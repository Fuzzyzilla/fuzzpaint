/// Utterly arbitrary :3
const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);
pub enum LogScope {
    /// A message from the remote server.
    /// # Warning
    /// It is important to make it clear to the user that this is an UNTRUSTED,
    /// ARBITRARY MESSAGE FROM THE REMOTE!
    Remote(ConnectionID),
    /// A message about the status of this connection.
    Regarding(ConnectionID),
    /// A message from the connections thread itself, not regarding any extant
    /// connection.
    Internal,
}

pub trait Waker: Sync {
    /// This should not block.
    fn wake(&mut self, which: ConnectionID);
    /// Recieve a log message from the connections daemon. See [`LogScope`] for
    /// the different kinds of messages. Default implementation forwards
    /// messages to the delivering thread's [`log::log!`].
    /// # Warning
    /// Beware the handling of [`LogScope::Remote`], see its docs for details.
    fn log(&mut self, scope: LogScope, level: log::Level, msg: &str) {
        match scope {
            LogScope::Internal => log::log!(level, "{msg}"),
            LogScope::Regarding(re) => log::log!(level, "[regarding {re:?}] {msg}"),
            // Use Dbg formatting for string to escape newlines (which can be
            // used maliciously). I wont pretend this is fool-proof.
            LogScope::Remote(re) => log::log!(level, "[untrusted message from {re:?}] {msg:?}"),
        }
    }
}

mod inner {
    use super::Waker;
    use fuzzpaint_connection::{client::Connection, tcp::client};

    pub struct Client {
        pub name: String,
        conn: client::Client,
        needs_flush: bool,
        pub messages: Vec<String>,
    }
    impl Client {
        pub fn defer_send(
            &mut self,
            message: &fuzzpaint_connection::client_msg::Message<'_>,
        ) -> std::io::Result<()> {
            self.conn.defer_send(message)?;
            self.needs_flush = true;
            Ok(())
        }
    }
    pub enum Request {
        /// Doesn't do anything, but wakes the thread.
        Poke,
        Exit,
        ConnectTcp(String, super::NewConnectionStatus),
    }

    pub struct Inner {
        pub clients: tokio::sync::Mutex<Vec<Client>>,
        // Acts semaphore with only one permit, but between async and sync contexts.
        pub exclusion: tokio::sync::Mutex<()>,
    }
    impl Inner {
        pub fn daemon(
            &self,
            mut requests: tokio::sync::mpsc::UnboundedReceiver<Request>,
            mut waker: Box<dyn Waker + Send>,
        ) -> anyhow::Result<()> {
            let rt = tokio::runtime::Builder::new_current_thread()
                // We dont do any blocking in the tokio RT, except that DNS
                // lookup is implemented as blocking internally. If more than
                // one DNS lookup is enqueued, they'll just happen sequentially.
                // This is fine uwu. Preferably we wouldn't have any auxiliary
                // threads...
                .max_blocking_threads(1)
                .thread_name_fn(|| {
                    #[cfg(debug_assertions)]
                    log::info!("tokio spawned an auxiliary thread");
                    "[rt] connection-poller".to_owned()
                })
                .enable_io()
                // Needed for connection timeout
                .enable_time()
                .build()?;
            let block = async {
                loop {
                    // Allows the controller thread to block the daemon at will.
                    drop(self.exclusion.lock().await);
                    let mut lock = self.clients.lock().await;

                    futures_util::future::join_all(lock.iter_mut().map(|client| async {
                        if client.needs_flush {
                            let res = client.conn.flush().await;
                            client.needs_flush = false;
                            Some(res)
                        } else {
                            None
                        }
                    }))
                    .await
                    .into_iter()
                    .for_each(|res| {
                        if let Some(res) = res {
                            res.expect("todo");
                        }
                    });
                    let fs =
                        lock.iter_mut()
                            .map(|client| async {
                                use fuzzpaint_connection::server_msg;
                                if let server_msg::Message {
                                    last_processed: (),
                                    message:
                                        server_msg::MessageKind::ServerMessage(
                                            server_msg::ServerMessage {
                                                user_id: _,
                                                message,
                                            },
                                        ),
                                } = client.conn.recv().await.expect("todo")
                                {
                                    client.messages.push(message.to_owned());
                                }
                            })
                            .collect::<Vec<_>>();
                    let race = crate::my_futures::race(fs);
                    let mut request = None;
                    tokio::select! {
                        biased;
                        // Bias towards requests that way "Exit" is handled
                        // eagerly.
                        new_request = requests.recv() => {
                            if let Some(new_request) = new_request {
                                request = Some(new_request);
                            } else {
                                return Ok(());
                            }
                        },
                        // Yields None if nothing to race. In that case, this
                        // fails to match, and the arm is disqualified.
                        Some(()) = race => {
                            waker.wake(crate::connections::ConnectionID(0));
                        },
                    };
                    drop(lock);
                    if let Some(request) = request {
                        match request {
                            Request::Poke => (),
                            Request::Exit => return Ok(()),
                            Request::ConnectTcp(addr, mut status) => {
                                // Cancel if it takes too long. Client::connect
                                // is not cancel safe, but this is ok.
                                match tokio::time::timeout(
                                    super::TIMEOUT,
                                    client::Client::connect(&addr),
                                )
                                .await
                                {
                                    Ok(Ok(client)) => {
                                        self.clients.lock().await.push(Client {
                                            name: addr,
                                            conn: client,
                                            messages: Vec::new(),
                                            needs_flush: false,
                                        });
                                        status.set_status(super::Pending::Success);
                                    }
                                    Ok(Err(e)) => {
                                        status.set_status(super::Pending::Fail);
                                        waker.log(
                                            crate::connections::LogScope::Internal,
                                            log::Level::Error,
                                            &format!("failed to connect to {addr}: {e}"),
                                        );
                                    }
                                    Err(_) => {
                                        status.set_status(super::Pending::Fail);
                                        waker.log(
                                            crate::connections::LogScope::Internal,
                                            log::Level::Error,
                                            &format!(
                                                "failed to connect to {addr}: connection timed out"
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            };
            rt.block_on(block)
        }
    }
}
use inner::{Inner, Request};

pub struct ClientConnectionsManager {
    inner: std::sync::Arc<Inner>,
    daemon: std::thread::JoinHandle<anyhow::Result<()>>,
    requests: tokio::sync::mpsc::UnboundedSender<Request>,
}
impl ClientConnectionsManager {
    /// Spawns the connection manager daemon, returning a handle to it.
    pub fn spawn(on_recv: Box<dyn Waker + Send>) -> anyhow::Result<Self> {
        let (send, recv) = tokio::sync::mpsc::unbounded_channel();
        let inner = std::sync::Arc::new(Inner {
            clients: Vec::new().into(),
            exclusion: ().into(),
        });
        let daemon = {
            let inner = inner.clone();
            std::thread::Builder::new()
                .name("connection-poller".to_owned())
                .spawn(move || inner.daemon(recv, on_recv))?
        };

        Ok(Self {
            inner,
            daemon,
            requests: send,
        })
    }
    pub fn connect_tcp(&self, address: String) {
        self.requests
            .send(Request::ConnectTcp(address, NewConnectionStatus::new()));
    }
    /// Kills the daemon, returning its result.
    /// # Panics
    /// If the daemon panicked.
    pub fn kill(self) -> anyhow::Result<()> {
        let _ = self.requests.send(Request::Exit);
        self.daemon
            .join()
            .unwrap_or_else(|e| std::panic::panic_any(e))
    }
    /// Get mutable access to all connections, may block.
    pub fn lock(&self) -> ConnectionsLock<'_> {
        // Prevent the daemon from continuing and immediately re-locking the
        // mutex we're trying to fetch.
        let _exclusion = self.inner.exclusion.blocking_lock();
        let _ = self.requests.send(Request::Poke);
        ConnectionsLock {
            clients: self.inner.clients.blocking_lock(),
            requests: self.requests.clone(),
        }
    }
}
pub struct ConnectionsLock<'a> {
    clients: tokio::sync::MutexGuard<'a, Vec<inner::Client>>,
    requests: tokio::sync::mpsc::UnboundedSender<Request>,
}
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConnectionID(usize);
pub struct ConnectionLock<'a> {
    client: &'a mut inner::Client,
}
impl ConnectionsLock<'_> {
    pub fn iter_connections(
        &'_ mut self,
    ) -> impl Iterator<Item = (ConnectionID, ConnectionLock<'_>)> {
        self.clients
            .iter_mut()
            .enumerate()
            .map(|(idx, client)| (ConnectionID(idx), ConnectionLock { client }))
    }
    pub fn connect(&self, address: String) -> NewConnectionStatus {
        let mut status = NewConnectionStatus::new();
        if self
            .requests
            .send(Request::ConnectTcp(address, status.private_clone()))
            .is_err()
        {
            status.set_status(Pending::Fail);
        }
        status
    }
}
impl ConnectionLock<'_> {
    pub fn name(&self) -> &str {
        &self.client.name
    }
    pub fn message<'m>(&'_ mut self, message: &'m str) -> std::io::Result<&'_ mut Self> {
        self.client
            .defer_send(&fuzzpaint_connection::client_msg::Message::ClientMessage(
                fuzzpaint_connection::client_msg::ClientMessage { message },
            ))?;
        Ok(self)
    }
    pub fn messages(&self) -> impl Iterator<Item = &str> {
        self.client.messages.iter().map(std::ops::Deref::deref)
    }
}
#[derive(PartialEq, Eq)]
#[repr(u8)]
enum Pending {
    Pending = 0,
    Success = 1,
    Fail = 2,
}
pub struct NewConnectionStatus {
    status: std::sync::Arc<std::sync::atomic::AtomicU8>,
}
impl NewConnectionStatus {
    fn new() -> Self {
        Self {
            status: std::sync::Arc::new((Pending::Pending as u8).into()),
        }
    }
    fn private_clone(&self) -> Self {
        Self {
            status: self.status.clone(),
        }
    }
    fn set_status(&mut self, status: Pending) {
        self.status
            .store(status as _, std::sync::atomic::Ordering::Relaxed);
    }
    fn get_status(&self) -> Pending {
        match self.status.load(std::sync::atomic::Ordering::Relaxed) {
            0 => Pending::Pending,
            1 => Pending::Success,
            _ => Pending::Fail,
        }
    }
    /// True if the connection succeeded. False if still pending.
    pub fn has_succeeded(&self) -> bool {
        self.get_status() == Pending::Success
    }
    /// True if the connection failed. False if still pending.
    pub fn has_failed(&self) -> bool {
        self.get_status() == Pending::Fail
    }
    /// Still waiting.
    pub fn is_pending(&self) -> bool {
        self.get_status() == Pending::Pending
    }
}

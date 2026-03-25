use std::{
    io::{Error, ErrorKind, Result},
    net::SocketAddr,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net,
};

const ONESHOT_ERROR: &str = "server in one-shot mode, no longer accepting connections";

#[derive(Default)]
pub struct Builder {
    oneshot: bool,
    allow_loopback_nodelay: bool,
}
impl Builder {
    pub fn oneshot(self, oneshot: bool) -> Self {
        Self { oneshot, ..self }
    }
    pub fn allow_loopback_nodelay(self, allow: bool) -> Self {
        Self {
            allow_loopback_nodelay: allow,
            ..self
        }
    }
    pub async fn bind_local(self) -> Result<Server> {
        use std::net::{Ipv4Addr, Ipv6Addr, SocketAddrV4, SocketAddrV6};
        self.bind(
            [
                SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::LOCALHOST, 0, 0, 0)),
                SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)),
            ]
            .as_slice(),
        )
        .await
    }
    pub async fn bind<A: net::ToSocketAddrs>(self, addr: A) -> Result<Server> {
        // Async because of DNS? woe.
        let listener = net::TcpListener::bind(addr).await?;
        Ok(Server {
            listener,
            oneshot: self.oneshot,
            allows_incoming: true.into(),
            allow_loopback_nodelay: self.allow_loopback_nodelay,
        })
    }
}

pub struct Server {
    listener: net::TcpListener,
    /// Stop accepting connections after the first successful connection.
    oneshot: bool,
    allows_incoming: std::sync::atomic::AtomicBool,
    allow_loopback_nodelay: bool,
}
impl Server {
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.listener.local_addr()
    }
}

impl crate::server::Connection for Server {
    type Error = Error;
    type Client = Client;
    fn allows_incoming(&self) -> bool {
        !self.oneshot
            || self
                .allows_incoming
                .load(std::sync::atomic::Ordering::Relaxed)
    }
    /// This function is *not* cancel safe. Cancelling the future may result in
    /// losing pending connections.
    async fn wait_client(&mut self) -> Result<Self::Client> {
        if self.oneshot && !self.allows_incoming() {
            return Err(Error::new(ErrorKind::NotConnected, ONESHOT_ERROR));
        }
        let (mut stream, address) = self.listener.accept().await?;
        if self.allow_loopback_nodelay && address.ip().is_loopback() {
            let _ = stream.set_nodelay(true);
        }

        let handshake = snow::Builder::new(super::noise_params())
            .prologue(super::PROTOCOL_VERSION)
            .unwrap()
            .build_responder()
            .map_err(Error::other)?;

        let (transport, session_id) =
            super::run_handshake(&mut stream, &mut Vec::new(), handshake).await?;

        if self.oneshot {
            // Successfully connected and one-shot mode, set the flag
            // disallowing further clients. May have changed since first check.
            let allow_connection = self
                .allows_incoming
                .swap(true, std::sync::atomic::Ordering::Relaxed);
            if !allow_connection {
                return Err(Error::new(ErrorKind::NotConnected, ONESHOT_ERROR));
            }
        }
        Ok(Client {
            stream,
            transport,
            session_id,
            address,
            buffer: bitcode::Buffer::new(),
            recv_staging: Vec::new(),
            send_staging: Vec::new(),
        })
    }
}
pub struct Client {
    stream: net::TcpStream,
    transport: snow::TransportState,
    session_id: Vec<u8>,
    address: SocketAddr,
    buffer: bitcode::Buffer,
    recv_staging: Vec<u8>,
    send_staging: Vec<u8>,
}
impl Client {
    /// Push a message to be sent on the next call to [`crate::server::ClientConnection::send`]
    pub fn defer_send(&mut self, message: &crate::server_msg::Message<'_>) -> Result<&mut Self> {
        super::encode_append(
            &mut self.buffer,
            &mut self.send_staging,
            &mut self.transport,
            message,
        )?;
        Ok(self)
    }
    /// Get the session ID. This is a cryptographic signature unique to this
    /// specific connection, and must be compared **out-of-band** with the
    /// server to ensure there is not a man-in-the-middle. This is not sensitive
    /// information.
    pub fn session_id(&self) -> String {
        super::long_hex_string(&self.session_id)
    }
}
impl crate::server::ClientConnection for Client {
    type Error = Error;
    /// Cancel-safe.
    async fn send(
        &mut self,
        message: &crate::server_msg::Message<'_>,
    ) -> std::result::Result<&mut Self, Self::Error> {
        super::streaming_write(
            &mut self.stream,
            &mut self.send_staging,
            &mut self.transport,
            &mut self.buffer,
            message,
        )
        .await?;
        Ok(self)
    }
    /// Cancel-safe.
    async fn flush(&mut self) -> std::result::Result<&mut Self, Self::Error> {
        self.stream.write_all(&self.send_staging).await?;
        self.send_staging.clear();
        self.stream.flush().await?;
        Ok(self)
    }
    /// Cancel-safe.
    async fn recv(&mut self) -> std::result::Result<crate::client_msg::Message<'_>, Self::Error> {
        super::streaming_read(
            &mut self.stream,
            &mut self.recv_staging,
            &mut self.transport,
            &mut self.buffer,
        )
        .await
    }
}

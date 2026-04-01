#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "server")]
pub mod server;
use tokio::sync::mpsc::{Receiver, Sender, error::SendError};

/// Fixme: use a byte-channel. Every message allocating is rough uwu BUT: This
/// is a sealed implementation detail, soooooooo doesn't matter rn. Heck, the
/// fact that the messages even get serialized a deserialized at all is an
/// implementation detail!
type Message = Box<[u8]>;

struct BidiBitcodeChannel {
    pub(super) send: Sender<Message>,
    pub(super) recv: Receiver<Message>,
    pub(super) recv_buffer: Box<[u8]>,
    pub(super) buffer: bitcode::Buffer,
}
impl BidiBitcodeChannel {
    async fn recv<'a, T: bitcode::Decode<'a>>(&'a mut self) -> Result<T, ConnectionError> {
        self.recv_buffer = self.recv.recv().await.ok_or(ConnectionError::Closed)?;
        self.buffer
            .decode(&self.recv_buffer)
            .map_err(ConnectionError::DecodeErr)
    }
    async fn send<T: bitcode::Encode>(&mut self, message: &T) -> Result<(), ConnectionError> {
        // Tests if the channel has closed before spending the work on encoding
        // the message.
        let permit = self
            .send
            .reserve()
            .await
            .map_err(|SendError(())| ConnectionError::Closed)?;
        let message = self.buffer.encode(message).into();
        permit.send(message);
        Ok(())
    }
    fn pair(capacity: usize) -> [Self; 2] {
        let a_to_b = tokio::sync::mpsc::channel(capacity);
        let b_to_a = tokio::sync::mpsc::channel(capacity);
        [
            BidiBitcodeChannel {
                send: a_to_b.0,
                recv: a_to_b.1,
                recv_buffer: Box::new([]),
                buffer: bitcode::Buffer::new(),
            },
            BidiBitcodeChannel {
                send: b_to_a.0,
                recv: b_to_a.1,
                recv_buffer: Box::new([]),
                buffer: bitcode::Buffer::new(),
            },
        ]
    }
}

#[derive(Debug)]
pub enum ConnectionError {
    Closed,
    DecodeErr(bitcode::Error),
}
impl std::fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectionError::Closed => write!(f, "channel closed"),
            ConnectionError::DecodeErr(e) => write!(f, "failed to decode message, {e}"),
        }
    }
}
impl std::error::Error for ConnectionError {}

#[cfg(all(feature = "server", feature = "client"))]
pub fn pair() -> (server::Server, client::Client) {
    let [server, client] = BidiBitcodeChannel::pair(64);
    (
        server::Server {
            takable_client: Some(server::Client { channel: server }),
        },
        client::Client { channel: client },
    )
}

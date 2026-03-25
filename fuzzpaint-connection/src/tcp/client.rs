use std::io::{Error, ErrorKind, Result};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net,
};
pub struct Client {
    stream: net::TcpStream,
    transport: snow::TransportState,
    session_id: Vec<u8>,
    buffer: bitcode::Buffer,
    // may contain residual bytes from half-recieved messages.
    recv_staging: Vec<u8>,
    // may contain residual bytes from half-sent messages.
    send_staging: Vec<u8>,
}
impl Client {
    /// This function is *not* cancel safe. Cancelling the future may result in
    /// a spurious connection to the address.
    pub async fn connect<A: net::ToSocketAddrs>(addr: A) -> Result<Self> {
        let mut stream = net::TcpStream::connect(addr).await?;

        let handshake = snow::Builder::new(super::noise_params())
            .prologue(super::PROTOCOL_VERSION)
            .unwrap()
            .build_initiator()
            .map_err(Error::other)?;

        let (transport, session_id) =
            super::run_handshake(&mut stream, &mut Vec::new(), handshake).await?;

        Ok(Self {
            stream,
            transport,
            session_id,
            send_staging: Vec::new(),
            recv_staging: Vec::new(),
            buffer: bitcode::Buffer::new(),
        })
    }
    /// Push a message to be sent on the next call to [`Connection::send`]
    pub fn defer_send(&mut self, message: &crate::client_msg::Message<'_>) -> Result<&mut Self> {
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
impl crate::client::Connection for Client {
    type Error = Error;
    /// Cancel-safe.
    async fn send(
        &mut self,
        message: &crate::client_msg::Message<'_>,
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
    async fn recv(&mut self) -> std::result::Result<crate::server_msg::Message<'_>, Self::Error> {
        super::streaming_read(
            &mut self.stream,
            &mut self.recv_staging,
            &mut self.transport,
            &mut self.buffer,
        )
        .await
    }
}

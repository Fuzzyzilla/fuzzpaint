use std::io::{Error, ErrorKind, Result};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net,
};
pub struct Client {
    stream: net::TcpStream,
    buffer: bitcode::Buffer,
    // may contain residual bytes from half-recieved messages.
    recv_staging: Vec<u8>,
    // may contain residual bytes from half-sent messages.
    send_staging: Vec<u8>,
}
impl Client {
    pub async fn connect<A: net::ToSocketAddrs>(addr: A) -> Result<Self> {
        let mut stream = net::TcpStream::connect(addr).await?;
        // "negotiate" a protocol "version"
        let mut protocol = [0u8; super::PROTOCOL_VERSION.len()];
        stream.read_exact(&mut protocol).await?;
        stream.write_all(super::PROTOCOL_VERSION).await?;
        if protocol != super::PROTOCOL_VERSION {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "server connected with unsupported protocol",
            ));
        }
        Ok(Self {
            stream,
            send_staging: Vec::new(),
            recv_staging: Vec::new(),
            buffer: bitcode::Buffer::new(),
        })
    }
}
impl crate::client::Connection for Client {
    type Error = Error;
    async fn send(
        &mut self,
        message: &crate::client_msg::Message<'_>,
    ) -> std::result::Result<&mut Self, Self::Error> {
        super::streaming_write(
            &mut self.stream,
            &mut self.buffer,
            &mut self.send_staging,
            message,
        )
        .await?;
        Ok(self)
    }
    async fn flush(&mut self) -> std::result::Result<&mut Self, Self::Error> {
        self.stream.write_all(&self.send_staging).await?;
        self.send_staging.clear();
        self.stream.flush().await?;
        Ok(self)
    }
    async fn recv(&mut self) -> std::result::Result<crate::server_msg::Message<'_>, Self::Error> {
        super::streaming_read(&mut self.stream, &mut self.buffer, &mut self.recv_staging).await
    }
}

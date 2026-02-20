use std::{
    io::{Error, ErrorKind, Read, Result, Write},
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
    pub fn bind_local(self) -> Result<Server> {
        self.bind(
            [
                net::SocketAddr::V6(net::SocketAddrV6::new(net::Ipv6Addr::LOCALHOST, 0, 0, 0)),
                net::SocketAddr::V4(net::SocketAddrV4::new(net::Ipv4Addr::LOCALHOST, 0)),
            ]
            .as_slice(),
        )
    }
    pub fn bind<A: net::ToSocketAddrs>(self, addr: A) -> Result<Server> {
        let listener = net::TcpListener::bind(addr)?;
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
    pub fn local_addr(&self) -> Result<net::SocketAddr> {
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
    fn wait_client(&self) -> Result<Self::Client> {
        if self.oneshot && !self.allows_incoming() {
            return Err(Error::new(ErrorKind::NotConnected, ONESHOT_ERROR));
        }
        let (mut stream, address) = self.listener.accept()?;
        if self.allow_loopback_nodelay && address.ip().is_loopback() {
            let _ = stream.set_nodelay(true);
        }
        // "Negotiate" a protocol version.
        // Tell the client, so that they may adjust to our maximum version.
        stream.write_all(super::PROTOCOL_VERSION)?;
        // Read the client's preferred version. May respond be an older version.
        let mut client_protocol_version = [0; super::PROTOCOL_VERSION.len()];
        stream.read_exact(&mut client_protocol_version)?;
        // Check if we can support this protocol version (trivial single-version
        // logic for now).
        if client_protocol_version != *super::PROTOCOL_VERSION {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "client connected with unsupported protocol",
            ));
        }
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
            address,
            buffer: bitcode::Buffer::new(),
            recv_staging: Vec::new(),
            send_staging: Vec::new(),
        })
    }
}
pub struct Client {
    stream: net::TcpStream,
    address: net::SocketAddr,
    buffer: bitcode::Buffer,
    recv_staging: Vec<u8>,
    send_staging: Vec<u8>,
}
impl crate::server::ClientConnection for Client {
    type Error = Error;
    fn send(
        &mut self,
        message: &crate::server_msg::Message<'_>,
    ) -> std::result::Result<(), Self::Error> {
        super::streaming_write(
            &mut self.stream,
            &mut self.buffer,
            &mut self.send_staging,
            message,
        )
    }
    fn flush(&mut self) -> std::result::Result<(), Self::Error> {
        self.stream.write_all(&self.send_staging)?;
        self.send_staging.clear();
        self.stream.flush()?;
        Ok(())
    }
    fn recv(&mut self) -> std::result::Result<crate::client_msg::Message<'_>, Self::Error> {
        super::streaming_read(&mut self.stream, &mut self.buffer, &mut self.recv_staging)
    }
}

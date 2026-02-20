pub trait Connection {
    type Error;
    type Client: ClientConnection;
    /// Returns whether the connection is still able to accept clients. E.g. for
    /// single-user servers, this will immediately become false after the first
    /// client connection. Returned value may be immediately out of date if
    /// connections are occuring on other threads.
    fn allows_incoming(&self) -> bool;
    fn wait_client(&self) -> Result<Self::Client, Self::Error>;
}
pub trait ClientConnection {
    type Error;
    fn send(&mut self, message: &crate::server_msg::Message<'_>) -> Result<(), Self::Error>;
    fn flush(&mut self) -> Result<(), Self::Error>;
    fn recv(&mut self) -> Result<crate::client_msg::Message<'_>, Self::Error>;
}
/// Represents a server with the ability to accept client(s).
pub struct Server<Conn: Connection> {
    conn: Conn,
}
/// From the server's side, represents a client of the server.
pub struct Client<Conn: Connection> {
    client: Conn::Client,
}

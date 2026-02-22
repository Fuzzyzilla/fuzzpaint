pub trait Connection {
    type Error: std::error::Error;
    /// Send a message to the server.
    async fn send(
        &mut self,
        message: &crate::client_msg::Message<'_>,
    ) -> Result<&mut Self, Self::Error>;
    /// Ensure all previous calls to [`Self::send`] have been executed.
    async fn flush(&mut self) -> Result<&mut Self, Self::Error>;
    /// Recieve a message to the server.
    async fn recv(&mut self) -> Result<crate::server_msg::Message<'_>, Self::Error>;
}
/// From the client side, represents a connection to a server.
struct Client<Conn: Connection> {
    conn: Conn,
}

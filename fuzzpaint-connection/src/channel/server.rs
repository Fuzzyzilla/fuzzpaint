pub struct Server {
    pub(super) takable_client: Option<Client>,
}
#[derive(Debug)]
pub enum WaitClientError {
    Consumed,
}
impl std::fmt::Display for WaitClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Consumed => write!(
                f,
                "server in one-shot mode, no longer accepting connections"
            ),
        }
    }
}
impl std::error::Error for WaitClientError {}
impl crate::server::Connection for Server {
    type Error = WaitClientError;
    type Client = Client;
    fn allows_incoming(&self) -> bool {
        self.takable_client.is_some()
    }
    async fn wait_client(&mut self) -> Result<Self::Client, Self::Error> {
        self.takable_client.take().ok_or(WaitClientError::Consumed)
    }
}
pub struct Client {
    pub(super) channel: super::BidiBitcodeChannel,
}
impl crate::server::ClientConnection for Client {
    type Error = super::ConnectionError;
    /// Cancel-safe.
    async fn send(
        &mut self,
        message: &crate::server_msg::Message<'_>,
    ) -> Result<&mut Self, Self::Error> {
        self.channel.send(message).await?;
        Ok(self)
    }
    /// Cancel-safe.
    async fn flush(&mut self) -> Result<&mut Self, Self::Error> {
        Ok(self)
    }
    /// Cancel-safe.
    async fn recv(&mut self) -> Result<crate::client_msg::Message<'_>, Self::Error> {
        self.channel.recv().await
    }
}

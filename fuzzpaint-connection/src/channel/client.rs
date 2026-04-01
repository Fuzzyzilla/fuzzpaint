pub struct Client {
    pub(super) channel: super::BidiBitcodeChannel,
}
impl crate::client::Connection for Client {
    type Error = super::ConnectionError;
    /// Cancel-safe.
    async fn send(
        &mut self,
        message: &crate::client_msg::Message<'_>,
    ) -> Result<&mut Self, Self::Error> {
        self.channel.send(message).await?;
        Ok(self)
    }
    /// Cancel-safe.
    async fn flush(&mut self) -> Result<&mut Self, Self::Error> {
        Ok(self)
    }
    /// Cancel-safe.
    async fn recv(&mut self) -> Result<crate::server_msg::Message<'_>, Self::Error> {
        self.channel.recv().await
    }
}

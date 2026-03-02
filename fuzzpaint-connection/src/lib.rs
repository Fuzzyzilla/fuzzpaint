//! # `fuzzpaint-connection`
//! Implements various wire protocols for connecting servers to clients as well
//! as higher-level interfaces for controlling remote documents. **This crate
//! does not expose a stable wire protocol,** all protocols are considered
//! implementation details and may change at any time (for now~).

#[cfg(feature = "channel")]
pub mod channel;
#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "tcp")]
pub mod tcp;

/// An ID, in the server's namespace. Refers to the same object across clients
/// and connections during the lifetime of the server.
pub type ID = ();
/// An ID, in a unique namespace for each direction of each client-server
/// connection. Reusable after discarded
pub type StreamID = ();
/// ID number of a client message.
pub type Serial = ();

#[macro_use]
pub mod macro_use {
    #[macro_export]
    macro_rules! message_enum {
    {
        $(#[$meta:meta])*
        pub enum $enum_name:ident$(<$($enum_lt:lifetime),+>)? {
            $(
                $name:ident$(<$($variant_lt:lifetime),+>)?
            ),*
            $(,)?
        }
    } => {
        $(#[$meta])*
        pub enum $enum_name$(<$($enum_lt,)*>)? {
            $(
                $name($name$(<$($variant_lt),*>)?)
            ),*
        }
    };
}
}

pub mod client_msg {
    use super::{ID, StreamID};

    super::message_enum!(
        #[cfg_attr(feature = "client", derive(bitcode::Encode))]
        #[cfg_attr(feature = "server", derive(bitcode::Decode))]
        pub enum Message<'a> {
            ClientMessage<'a>,
            //ClientHello<'a>,
        }
    );

    /// Initial message sent from client to server to introduce itself
    #[derive(bitcode::Decode)]
    #[cfg_attr(feature = "client", derive(bitcode::Encode))]
    pub struct ClientHello<'a> {
        pub software: &'a str,
        pub username: &'a str,
        pub color: [u8; 3],
    }
    #[cfg_attr(feature = "client", derive(bitcode::Encode))]
    #[cfg_attr(feature = "server", derive(bitcode::Decode))]
    pub struct ClientMessage<'a> {
        pub message: &'a str,
    }
    #[cfg_attr(feature = "client", derive(bitcode::Encode))]
    #[cfg_attr(feature = "server", derive(bitcode::Decode))]
    struct RequestDocument {
        document: ID,
    }
    #[cfg_attr(feature = "client", derive(bitcode::Encode))]
    #[cfg_attr(feature = "server", derive(bitcode::Decode))]
    struct Motion {
        position: Option<(f32, f32)>,
    }
    #[cfg_attr(feature = "client", derive(bitcode::Encode))]
    #[cfg_attr(feature = "server", derive(bitcode::Decode))]
    struct BeginStroke {
        name: StreamID,
        aspects: u16,
    }
    #[cfg_attr(feature = "client", derive(bitcode::Encode))]
    #[cfg_attr(feature = "server", derive(bitcode::Decode))]
    struct InlineBlob {
        name: StreamID,
        // FIXME: Why doesn't byte slice work here? &str works...
        data: Vec<u8>,
        finish: bool,
    }
}
pub mod server_msg {
    use super::{ID, Serial, StreamID};

    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    pub struct Message<'a> {
        pub last_processed: Serial,
        pub message: MessageKind<'a>,
    }
    super::message_enum!(
        #[cfg_attr(feature = "server", derive(bitcode::Encode))]
        #[cfg_attr(feature = "client", derive(bitcode::Decode))]
        pub enum MessageKind<'a> {
            ServerMessage<'a>,
            //Log<'a>,
            //Error,
        }
    );

    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    enum ErrorKind {
        UnknownID(ID),
        UnknownStreamID(StreamID),
        PermissionDenied,
        EnhanceYourChill,
    }
    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    struct Error {
        message: Serial,
        kind: ErrorKind,
    }

    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    pub enum LogLevel {
        Warn,
        Error,
        Info,
        Debug,
        Trace,
    }
    impl From<log::Level> for LogLevel {
        fn from(value: log::Level) -> Self {
            match value {
                log::Level::Debug => Self::Debug,
                log::Level::Error => Self::Error,
                log::Level::Info => Self::Info,
                log::Level::Trace => Self::Trace,
                log::Level::Warn => Self::Warn,
            }
        }
    }
    impl From<LogLevel> for log::Level {
        fn from(value: LogLevel) -> Self {
            match value {
                LogLevel::Debug => Self::Debug,
                LogLevel::Error => Self::Error,
                LogLevel::Info => Self::Info,
                LogLevel::Trace => Self::Trace,
                LogLevel::Warn => Self::Warn,
            }
        }
    }
    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    struct Log<'a> {
        level: LogLevel,
        text: &'a str,
    }
    /// Initial message sent from server to client to introduce itself
    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    pub struct ServerHello<'a> {
        pub software: &'a str,
        pub motd: &'a str,
    }
    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    pub struct ServerMessage<'a> {
        pub user_id: Option<ID>,
        pub message: &'a str,
    }

    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    pub struct AdvertiseOtherUser<'a> {
        pub id: ID,
        pub hello: super::client_msg::ClientHello<'a>,
    }
    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    pub struct OtherUserRemoved {
        pub id: ID,
    }
    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    pub struct AdvertiseDocument<'a> {
        pub id: ID,
        pub name: Option<&'a str>,
    }
    #[cfg_attr(feature = "server", derive(bitcode::Encode))]
    #[cfg_attr(feature = "client", derive(bitcode::Decode))]
    pub struct DocumentRemoved {
        pub id: ID,
    }
}

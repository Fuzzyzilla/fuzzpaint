#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "server")]
pub mod server;

const PROTOCOL_VERSION: &[u8] = b"owo lmao hai this is a fuzzpaint connection";

fn streaming_read<'a, T: bitcode::Decode<'a>>(
    mut stream: impl std::io::Read,
    buffer: &mut bitcode::Buffer,
    staging: &'a mut Vec<u8>,
) -> std::io::Result<T> {
    use std::io::{Error, ErrorKind};
    // Clean up previous recv, if was long enough to parse.
    if let &[a, b, ..] = staging.as_slice() {
        let reported_length = usize::from(u16::from_le_bytes([a, b]));
        if staging.len() >= reported_length + 2 {
            staging.drain(..reported_length + 2);
        }
    }
    // Read more data in a loop until long enough to parse
    let message_length = loop {
        let grow_by = if let &[a, b, ..] = staging.as_slice() {
            let reported_length = usize::from(u16::from_le_bytes([a, b]));
            // There's already a whole message in here!!
            if staging.len() >= reported_length + 2 {
                break reported_length;
            }
            let grow_by = reported_length - staging.len() + 2;

            // Ensure we don't get caught in an infinite loop (grow_by = 0)
            // or get overzealous with our allocation. reported_length is
            // untrusted data, remember~
            grow_by.clamp(32, 256)
        } else {
            // Not enough data to know.
            128
        };
        staging.reserve(grow_by);
        let spare = staging.spare_capacity_mut();
        // Infinite loop!!!! prevented above.
        debug_assert!(!spare.is_empty());
        // Read accepts ref to bytes, so they must be valid (i.e. init)
        // bytes. FIXME: this is redundant most of the time uwu, as it just
        // repeatedly zeros already-init bytes.
        spare.fill(std::mem::MaybeUninit::zeroed());
        let spare = unsafe { spare.assume_init_mut() };

        let read = stream.read(spare);
        let read = match read {
            Ok(read) => read,
            Err(e) => {
                if e.kind() == ErrorKind::Interrupted {
                    continue;
                } else {
                    return Err(e);
                }
            }
        };
        if read == 0 {
            // closed. We know that there's no residual complete messages,
            // so error out.
            return Err(Error::new(ErrorKind::NotConnected, "connection closed"));
        }
        let new_len = staging.len() + read;
        // This is against the contract of Read, but, being a safe
        // trait, we mustn't make unsafe assertions based on it's
        // contracts.
        if new_len > staging.capacity() {
            return Err(Error::other("read reported invalid length"));
        }
        unsafe {
            // Even if `read` didn't actually write these bytes (it may not
            // have!) they've been safely zeroed above and so can be assumed
            // valid.
            staging.set_len(new_len);
        }
        let &[a, b, ..] = staging.as_slice() else {
            continue;
        };
        let message_length = usize::from(u16::from_le_bytes([a, b]));
        // Ready to attempt decode?
        if staging.len() >= message_length + 2 {
            break message_length;
        }
    };
    let Some(data) = staging.get(2..2 + message_length) else {
        // Loop above only breaks when this codition is met. Can't break
        // with the data slice to make this infallible, for some inscrutable
        // lifetime reason :o
        return Err(Error::other("unreachable"));
    };
    buffer
        .decode(data)
        .map_err(|e| Error::new(ErrorKind::InvalidData, e))
}
fn streaming_write<T: bitcode::Encode>(
    mut stream: impl std::io::Write,
    buffer: &mut bitcode::Buffer,
    staging: &mut Vec<u8>,
    t: &T,
) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    let bytes = buffer.encode(t);
    // Bitcode doesn't know the length of it's own messages (slices passed
    // to decode must contain exactly the correct number of bytes). So, we
    // must prefix each message with a length on the wire-level.
    staging.reserve(bytes.len() + 2);
    // cant use usize here, since that differs in length by platform.
    let len = u16::try_from(bytes.len())
        .map_err(|_| Error::new(ErrorKind::InvalidData, "message too long"))?;
    staging.extend_from_slice(&len.to_le_bytes());
    staging.extend_from_slice(bytes);

    // Send as much as we can, buffer the rest for later.
    let sent = stream.write(staging)?;
    if sent == 0 {
        // 0 = unlikely to ever accept bytes again
        return Err(Error::new(ErrorKind::NotConnected, "connection closed"));
    }
    staging.drain(..sent.min(staging.len()));
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        client::Connection as _,
        server::{ClientConnection as _, Connection as _},
    };
    use std::io::Result;
    #[cfg(all(feature = "client", feature = "server"))]
    #[test]
    fn wawa() -> Result<()> {
        let server = server::Builder::default().bind_local()?;
        let address = server.local_addr()?;

        let thread = std::thread::Builder::new()
            .spawn(move || -> std::io::Result<()> {
                let mut client = client::Client::connect(address)?;
                let message = client.recv()?;
                assert!(matches!(
                    message,
                    crate::server_msg::Message {
                        last_processed: (),
                        message: crate::server_msg::MessageKind::ServerMessage(
                            crate::server_msg::ServerMessage {
                                user_id: None,
                                message: "hai from server :3",
                            },
                        ),
                    }
                ));
                client.send(&crate::client_msg::Message::ClientMessage(
                    crate::client_msg::ClientMessage {
                        message: "hello in turn ;3",
                    },
                ))?;
                client.flush()?;

                Ok(())
            })
            .unwrap();

        let mut client = server.wait_client()?;

        client.send(&crate::server_msg::Message {
            last_processed: (),
            message: crate::server_msg::MessageKind::ServerMessage(
                crate::server_msg::ServerMessage {
                    user_id: None,
                    message: "hai from server :3",
                },
            ),
        })?;
        client.flush()?;
        let message = client.recv()?;
        assert!(matches!(
            message,
            crate::client_msg::Message::ClientMessage(crate::client_msg::ClientMessage {
                message: "hello in turn ;3",
            },)
        ));

        thread.join().unwrap()?;
        Ok(())
    }
}

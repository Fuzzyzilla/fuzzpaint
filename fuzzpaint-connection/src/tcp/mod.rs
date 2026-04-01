#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "server")]
pub mod server;

const MAX_BODY_LENGTH: usize = 65535;
const SIZE_PREFIX_LENGTH: usize = 2;
const MAX_LENGTH_WITH_PREFIX: usize = MAX_BODY_LENGTH + SIZE_PREFIX_LENGTH;

/// Format a slice of data as a really long hex number.
fn long_hex_string(data: &[u8]) -> String {
    // Format as hex, 2 chars per byte.
    let mut string = String::with_capacity(data.len() * 2);
    for byte in data {
        let high = byte >> 4;
        let low = byte & 15;
        string.push(char::from_digit(high.into(), 16).unwrap());
        string.push(char::from_digit(low.into(), 16).unwrap());
    }
    string
}

/// Grow the staging buffer little by little, up to MAX_LENGTH_WITH_PREFIX.
fn grow_stage(staging: &mut Vec<u8>) {
    // We have no idea the size here. But the vec interally grows exponentially,
    // so this is okay~.
    staging.reserve(128);
    // Expand to fill capacity, up to MAX_LENGTH_WITH_PREFIX.
    if staging.len() < staging.capacity() && staging.len() < MAX_LENGTH_WITH_PREFIX {
        // We only need up to MAX_LENGTH_WITH_PREFIX bytes.
        let added_bytes = staging.capacity().min(MAX_LENGTH_WITH_PREFIX) - staging.len();
        let new_length = staging.len() + added_bytes;
        // Assert we aren't doing a buffer overrun
        assert!(new_length <= staging.capacity());
        // Fill those bytes with zeros...
        staging.spare_capacity_mut()[..added_bytes].fill(std::mem::MaybeUninit::zeroed());
        // Safety: We just initted them, and the assert above proves we aren't
        // setting len > cap.
        unsafe { staging.set_len(new_length) };
    }
}
/// This is an implimentation detail! This crate purposefully does not define a
/// protocol.
fn noise_params() -> snow::params::NoiseParams {
    // Noise: Use noise protocol.
    // NN: No static key on either end.
    // 25519: Use eliptic curve 25519 for Diffie-Hellman negotiation.
    // ChaChaPoly: Use the ChaChaPoly cipher.
    // BLAKE2b: Use the Blake2b cipher (optimized for 64bit).

    // In order to change any of these, the features in the Cargo.toml should be
    // updated to match.
    "Noise_NN_25519_ChaChaPoly_BLAKE2b"
        .parse()
        .expect("compiled without necessary encryption engines")
}
/// Run a responder or initiator to completion. Returns the prepared transport
/// state, as well as an opaque, but generally human-readable, session-unique ID
/// for out-of-band verifying against man-in-the-middle attacks.
///
/// The `staging` buffer is used as scratch space to store messages during
/// encoding.
async fn run_handshake(
    mut stream: impl tokio::io::AsyncWriteExt + Unpin + tokio::io::AsyncReadExt,
    staging: &mut Vec<u8>,
    mut handshake: snow::HandshakeState,
) -> std::io::Result<(snow::TransportState, Vec<u8>)> {
    const NOISE_MAX_LENGTH: usize = 65535;
    const PREFIX_LENGTH: usize = 2;
    const MAX_LENGTH_WITH_PREFIX: usize = NOISE_MAX_LENGTH + PREFIX_LENGTH;
    use std::io::Error;

    // Staging should be clear at every direction transition (send -> recv, and
    // recv -> send).
    staging.clear();
    let mut was_writing = false;

    loop {
        // We're done here, make the transport state and return.
        if handshake.is_handshake_finished() {
            let hash = handshake.get_handshake_hash().to_owned();
            return match handshake.into_transport_mode() {
                Ok(transport) => Ok((transport, hash)),
                Err(e) => Err(Error::other(e)),
            };
        }
        // My turn to *send*?
        if handshake.is_my_turn() {
            // Transitioning into write, prepare!
            if !was_writing {
                if !staging.is_empty() {
                    // The reading process left residual bytes. The remote tried
                    // to send a message even though the state machine says its
                    // our turn. Oh no!
                    return Err(Error::other("remote sent bytes while not its turn"));
                }
                // Init it!
                grow_stage(staging);
            }
            was_writing = true;

            let Some((size_prefix, body)) = staging.split_at_mut_checked(SIZE_PREFIX_LENGTH) else {
                // There's no body space to read into, we need to allocate some.
                grow_stage(staging);
                // Retry.
                continue;
            };
            match handshake.write_message(&[], body) {
                Ok(body_len) => {
                    let len_with_prefix = body_len + SIZE_PREFIX_LENGTH;

                    // Write the length prefix
                    let body_len: u16 = body_len
                        .try_into()
                        .map_err(|_| Error::other("invalid encrypted length"))?;
                    size_prefix.copy_from_slice(&body_len.to_le_bytes());

                    // Send the whole message, prefix and body.
                    let Some(message) = staging.get(..len_with_prefix) else {
                        return Err(Error::other("invalid encrypted length"));
                    };
                    stream.write_all(message).await?;
                    // Prepare for direction swap.
                    staging.clear();
                }
                // Not enough staging space AND we have space to grow.
                Err(snow::Error::Input) if staging.len() < MAX_LENGTH_WITH_PREFIX => {
                    grow_stage(staging);
                    // Loop falls through and tries again.
                }
                // Unrecoverable error.
                Err(e) => return Err(Error::other(e)),
            }
        } else {
            was_writing = false;
            // True iff there's a whole message within the staging buffer.
            if let &[low, high, ref body @ ..] = &staging[..]
                && usize::from(u16::from_le_bytes([low, high])) <= body.len()
            {
                // Has a whole message! Decode it!
                let body_len = usize::from(u16::from_le_bytes([low, high]));

                // Returns payload size, always zero (asserted by the empty
                // payload unpack buffer.)
                let always_zero = handshake
                    .read_message(&body[..body_len], &mut [])
                    .map_err(Error::other)?;
                debug_assert_eq!(always_zero, 0);

                // Consume the message
                staging.drain(..body_len + SIZE_PREFIX_LENGTH);
            } else {
                // A full message has not yet been recieved, read more.
                stream.read_buf(staging).await?;
                // Loop falls through and tries again.
            };
        }
    }
}

const PROTOCOL_VERSION: &[u8] = b"owo lmao hai this is a fuzzpaint connection";

async fn streaming_read<'a, T: bitcode::Decode<'a>>(
    mut stream: impl tokio::io::AsyncReadExt + Unpin,
    staging: &'a mut Vec<u8>,
    transport: &mut snow::TransportState,
    buffer: &mut bitcode::Buffer,
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

        let read = stream.read(spare).await;
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
fn encode_append<T: bitcode::Encode>(
    buffer: &mut bitcode::Buffer,
    staging: &mut Vec<u8>,
    transport: &mut snow::TransportState,
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

    Ok(())
}
async fn streaming_write<T: bitcode::Encode>(
    mut stream: impl tokio::io::AsyncWriteExt + Unpin,
    staging: &mut Vec<u8>,
    transport: &mut snow::TransportState,
    buffer: &mut bitcode::Buffer,
    t: &T,
) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    encode_append(buffer, staging, transport, t)?;

    // Send as much as we can, buffer the rest for later.
    let sent = stream.write(staging).await?;
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
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .build()?;
        let server = rt.block_on(server::Builder::default().bind_local())?;
        let addr = server.local_addr()?;
        let (client, server) = rt.block_on(async { tokio::join!(client(addr), serve(server)) });
        return client.and(server);

        async fn serve(mut server: server::Server) -> Result<()> {
            let mut client = server.wait_client().await?;

            client
                .send(&crate::server_msg::Message {
                    last_processed: (),
                    message: crate::server_msg::MessageKind::ServerMessage(
                        crate::server_msg::ServerMessage {
                            user_id: None,
                            message: "hai from server :3",
                        },
                    ),
                })
                .await?
                .flush()
                .await?;
            let message = client.recv().await?;
            assert!(matches!(
                message,
                crate::client_msg::Message::ClientMessage(crate::client_msg::ClientMessage {
                    message: "hello in turn ;3",
                },)
            ));
            Ok(())
        }
        async fn client(addr: std::net::SocketAddr) -> Result<()> {
            let mut client = client::Client::connect(addr).await?;
            let message = client.recv().await?;
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
            client
                .send(&crate::client_msg::Message::ClientMessage(
                    crate::client_msg::ClientMessage {
                        message: "hello in turn ;3",
                    },
                ))
                .await?
                .flush()
                .await?;

            Ok(())
        }
    }
}

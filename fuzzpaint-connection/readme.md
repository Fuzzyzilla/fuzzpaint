# Fuzzpaint Connection

The low-level connection between a Fuzzpaint client and server, for several
configurations thereof.

Transports (Cargo feature names match):
* `tcp`: Networked connection on top of TCP.
* `channel`: Simple Channel-based connection to a server that lives within the
  process. Zerocopy shared memory is possible, but the server and client are not
  insulated from each other.
* `ipc-shm`: Interprocess connection. Unidirectional zerocopy shared memory is
  possible.
  * Unidirectionality of the shared memory is chosen to insulate the user's
    data, so that the external server process *cannot* be negatively affected by
    anything the client does while still taking advantage of the performance
    benefits of directly sharing memory.

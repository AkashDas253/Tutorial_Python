## Python `socket` Module 

### Overview

The `socket` module provides low-level network communication capabilities in Python, enabling communication between devices via TCP, UDP, and other protocols. It is a wrapper around the BSD socket interface.

---

## **Core Concepts**

### **Socket Types**

* **Stream sockets** (`SOCK_STREAM`)

  * TCP-based, reliable, connection-oriented.
* **Datagram sockets** (`SOCK_DGRAM`)

  * UDP-based, connectionless, faster but less reliable.
* **Raw sockets** (`SOCK_RAW`)

  * Low-level network protocols; require admin/root privileges.

### **Address Families**

* `AF_INET` → IPv4
* `AF_INET6` → IPv6
* `AF_UNIX` → Local inter-process communication (Unix domain sockets)
* `AF_BLUETOOTH` → Bluetooth sockets

### **Socket Creation**

```python
socket.socket(
    family=AF_INET,       # Address family
    type=SOCK_STREAM,     # Socket type
    proto=0               # Protocol (0 = default for given family and type)
)
```

### **Socket Lifecycle**

1. **Server Side**

   * `socket()` → Create a socket.
   * `bind()` → Bind to IP and port.
   * `listen()` → Start listening for connections (TCP).
   * `accept()` → Accept incoming connection.
   * `recv()` / `send()` → Exchange data.
   * `close()` → Close connection.
2. **Client Side**

   * `socket()` → Create a socket.
   * `connect()` → Connect to server.
   * `send()` / `recv()` → Exchange data.
   * `close()` → Close connection.

---

## **Key Functions & Methods**

| Method / Function        | Description                                                               |
| ------------------------ | ------------------------------------------------------------------------- |
| `socket()`               | Create a new socket.                                                      |
| `bind(address)`          | Bind socket to `(host, port)` or a path.                                  |
| `listen([backlog])`      | Enable server to accept connections; `backlog` is max queued connections. |
| `accept()`               | Accept a connection, returns `(conn, address)`.                           |
| `connect(address)`       | Connect to a remote socket.                                               |
| `connect_ex(address)`    | Like `connect()` but returns error code instead of raising exception.     |
| `send(bytes)`            | Send data to connected socket.                                            |
| `sendall(bytes)`         | Send all data, retrying as necessary.                                     |
| `recv(bufsize)`          | Receive up to `bufsize` bytes.                                            |
| `sendto(bytes, address)` | Send data to a specific address (UDP).                                    |
| `recvfrom(bufsize)`      | Receive data and sender address (UDP).                                    |
| `settimeout(timeout)`    | Set blocking timeout.                                                     |
| `gettimeout()`           | Get current timeout.                                                      |
| `setblocking(flag)`      | Set blocking or non-blocking mode.                                        |
| `shutdown(how)`          | Shut down part of the connection (`SHUT_RD`, `SHUT_WR`, `SHUT_RDWR`).     |
| `close()`                | Close socket.                                                             |
| `getsockname()`          | Get local address.                                                        |
| `getpeername()`          | Get remote address.                                                       |
| `fileno()`               | File descriptor of socket.                                                |

---

## **Constants**

* Address Families: `AF_INET`, `AF_INET6`, `AF_UNIX`, `AF_BLUETOOTH`
* Socket Types: `SOCK_STREAM`, `SOCK_DGRAM`, `SOCK_RAW`, `SOCK_RDM`, `SOCK_SEQPACKET`
* Protocols: `IPPROTO_TCP`, `IPPROTO_UDP`, etc.
* Shutdown Modes: `SHUT_RD`, `SHUT_WR`, `SHUT_RDWR`

---

## **Error Handling**

* Exceptions are subclasses of `OSError`.
* Common:

  * `socket.error`
  * `socket.timeout`
  * `socket.gaierror` (getaddrinfo failures)
  * `socket.herror` (host-related errors)

---

## **Advanced Features**

* **Non-blocking I/O**: With `setblocking(False)` or `settimeout(0)`
* **Socket options** via `setsockopt(level, optname, value)`

  * Examples: `SO_REUSEADDR`, `SO_KEEPALIVE`, `TCP_NODELAY`
* **File-like objects** via `makefile()` (allows using `.read()`/`.write()`)
* **IPv6 support**
* **Select / Poll** integration for multiplexing

---

## **Usage Scenarios**

* Web servers and clients (HTTP, custom protocols)
* Chat applications
* IoT device communication
* Multiplayer games
* Inter-process communication (IPC)
* Network monitoring tools

---

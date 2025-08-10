## WebSockets in Flask  

### Overview  
WebSockets allow real-time, bi-directional communication between a client and a server. Flask does not support WebSockets natively, but **Flask-SocketIO** provides this functionality using **WebSockets**, **long polling**, and **other transports**.

---

## Installation  
```sh
pip install flask-socketio
```

---

## Setting Up Flask-SocketIO  
```python
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return "WebSocket Server Running"

if __name__ == '__main__':
    socketio.run(app, debug=True)
```

---

## Handling WebSocket Events  

### Server-side (Flask)  
```python
@socketio.on('message')
def handle_message(msg):
    print(f"Received message: {msg}")
    socketio.send(f"Server received: {msg}")

@socketio.on('custom_event')
def handle_custom_event(data):
    print(f"Received data: {data}")
    socketio.emit('response_event', {"message": "Processed"})
```

---

### Client-side (JavaScript)  
```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
    console.log("Connected to WebSocket server");
    socket.send("Hello, Server!");
});

socket.on('message', (data) => {
    console.log("Server says:", data);
});

socket.emit('custom_event', { user: "Alice", action: "joined" });

socket.on('response_event', (data) => {
    console.log("Custom event response:", data);
});
```

---

## Broadcasting Messages  
Send messages to **all clients**:  
```python
@socketio.on('broadcast')
def handle_broadcast(data):
    socketio.emit('broadcast_message', data)
```

---

## Rooms (Group Communication)  
```python
from flask_socketio import join_room, leave_room

@socketio.on('join')
def handle_join(data):
    room = data["room"]
    join_room(room)
    socketio.emit('room_message', f"User joined {room}", room=room)

@socketio.on('leave')
def handle_leave(data):
    room = data["room"]
    leave_room(room)
    socketio.emit('room_message', f"User left {room}", room=room)
```

---

## Summary  

| Feature | Description |
|---------|------------|
| **Basic Setup** | `Flask-SocketIO` enables WebSockets |
| **Event Handling** | `@socketio.on('event')` for real-time communication |
| **Broadcasting** | `socketio.emit('event', data)` to all clients |
| **Rooms** | `join_room(room)`, `leave_room(room)` for group messaging |

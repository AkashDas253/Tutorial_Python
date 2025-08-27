## Gunicorn 

### What is Gunicorn?

**Gunicorn (Green Unicorn)** is a **Python WSGI HTTP server** for UNIX. It acts as an **interface between web servers (like Nginx)** and **Python web applications** (like Django).

* **WSGI-compatible**
* **Lightweight**, **pre-fork worker model**
* Used in **production** deployments of Django apps

---

### Key Features

* WSGI-compliant: works with Django, Flask, etc.
* Multiple worker types (sync, async)
* Pre-fork architecture (like Unicorn in Ruby)
* Easily integrates with **Nginx**, **systemd**, or **supervisord**

---

### Installation

```bash
pip install gunicorn
```

---

### Running Django with Gunicorn

```bash
gunicorn myproject.wsgi:application
```

You can add parameters:

```bash
gunicorn myproject.wsgi:application --bind 0.0.0.0:8000 --workers 3
```

| Parameter     | Description                                |
| ------------- | ------------------------------------------ |
| `--bind`      | IP and port to listen on                   |
| `--workers`   | Number of worker processes (2 × CPUs + 1)  |
| `--timeout`   | Time (sec) before killing unresponsive req |
| `--log-level` | Log level: `debug`, `info`, `warning`, etc |

---

### Worker Types

| Type                            | Use Case                                    |
| ------------------------------- | ------------------------------------------- |
| `sync`                          | Default, simple, handles one req per worker |
| `eventlet`                      | For async I/O, requires eventlet lib        |
| `gevent`                        | For async + green threads                   |
| `tornado`                       | Use Tornado IOLoop                          |
| `gthread`                       | Thread workers                              |
| `uvicorn.workers.UvicornWorker` | For ASGI (e.g. FastAPI)                     |

```bash
gunicorn myproject.wsgi:application -k gevent
```

---

### Gunicorn with Nginx

Typically, you’ll put Gunicorn **behind Nginx**:

**Nginx** → **Gunicorn** → **Django (WSGI)**

* Nginx handles static files, HTTPS, reverse proxy
* Gunicorn handles WSGI requests

---

### Systemd Integration (Linux)

Create a service file:

```ini
# /etc/systemd/system/gunicorn.service
[Unit]
Description=gunicorn daemon
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/myproject
ExecStart=/home/ubuntu/venv/bin/gunicorn myproject.wsgi:application \
          --bind 127.0.0.1:8000 --workers 3

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable gunicorn
sudo systemctl start gunicorn
```

---

### Logging

```bash
gunicorn myproject.wsgi:application \
  --access-logfile access.log \
  --error-logfile error.log
```

---

### Production Notes

* Use **`--daemon`** to run Gunicorn in background (not recommended with systemd)
* Use **supervisor** or **systemd** to manage processes
* Gunicorn does **not serve static files** – use Nginx or WhiteNoise for that

---

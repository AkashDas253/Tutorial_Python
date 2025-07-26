## uWSGI 

### What is uWSGI?

**uWSGI** is a **full-stack application server** for deploying Python web apps, implementing the **WSGI (Web Server Gateway Interface)** standard.

* Supports WSGI, CGI, FastCGI, ASGI, HTTP, and more
* Often used with **Nginx** in production
* Known for **performance, flexibility, and configuration depth**

---

### uWSGI vs Gunicorn

| Feature       | uWSGI                                 | Gunicorn                      |
| ------------- | ------------------------------------- | ----------------------------- |
| Language      | C                                     | Python                        |
| Protocols     | WSGI, uwsgi, HTTP, FastCGI            | WSGI only                     |
| Configuration | INI/XML/Command-line                  | Mostly CLI-based              |
| Plugin-based  | Yes (highly extensible)               | No                            |
| Async support | Yes (via plugins, limited)            | Limited (via gevent/eventlet) |
| Usage         | More powerful, steeper learning curve | Simpler, plug-and-play        |

---

### Installation

```bash
pip install uwsgi
```

---

### Running Django with uWSGI

```bash
uwsgi --http :8000 --module myproject.wsgi
```

Add workers, threads, and other options:

```bash
uwsgi --http :8000 \
      --module myproject.wsgi \
      --workers 4 \
      --threads 2
```

---

### Common uWSGI CLI Options

| Option             | Description                              |
| ------------------ | ---------------------------------------- |
| `--http`           | Serve via HTTP (for dev only)            |
| `--socket`         | Serve via UNIX socket or TCP/IP          |
| `--module`         | WSGI entry point (e.g. `myproject.wsgi`) |
| `--master`         | Enable master process                    |
| `--processes`      | Number of worker processes               |
| `--threads`        | Threads per worker                       |
| `--vacuum`         | Clean up sockets/pidfiles after exit     |
| `--daemonize`      | Log output to file                       |
| `--enable-threads` | Enable thread support                    |

---

### Using an INI File (Recommended for Production)

**`uwsgi.ini`**

```ini
[uwsgi]
chdir = /path/to/your/project
module = myproject.wsgi:application
master = true
processes = 4
threads = 2
socket = /run/uwsgi/myproject.sock
chmod-socket = 660
vacuum = true
die-on-term = true
```

Then run with:

```bash
uwsgi --ini uwsgi.ini
```

---

### uWSGI + Nginx (Typical Production Stack)

**Nginx** serves static files and proxies requests to **uWSGI socket**.

**Nginx config:**

```nginx
location / {
    include uwsgi_params;
    uwsgi_pass unix:/run/uwsgi/myproject.sock;
}
```

---

### uWSGI Emperor Mode (Multiple App Management)

Used to manage multiple apps with one master uWSGI process (the “Emperor”).

```bash
uwsgi --emperor /etc/uwsgi/vassals --uid www-data --gid www-data
```

Each app gets its own `.ini` file in the `vassals` folder.

---

### Systemd Service File

```ini
# /etc/systemd/system/uwsgi.service
[Unit]
Description=uWSGI Emperor
After=network.target

[Service]
ExecStart=/usr/local/bin/uwsgi --emperor /etc/uwsgi/vassals
Restart=always
KillSignal=SIGQUIT
Type=notify
NotifyAccess=all

[Install]
WantedBy=multi-user.target
```

---

### Logs

```bash
uwsgi --ini uwsgi.ini --daemonize /var/log/uwsgi.log
```

---

### Notes for Django

* Use `--static-map` or **WhiteNoise** for static files if not using Nginx
* For **ASGI/Django Channels**, use **Uvicorn/Daphne**, not uWSGI
* Use `.sock` file for better performance with Nginx than plain HTTP

---

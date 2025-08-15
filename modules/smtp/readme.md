## `smtplib` (SMTP Client for Sending Emails in Python)

### Purpose

* The `smtplib` module defines an SMTP client session object for sending emails using the **Simple Mail Transfer Protocol**.
* Can send plain text, HTML emails, or emails with attachments.
* Supports both unencrypted and encrypted connections (TLS/SSL).

---

### Key Classes and Methods

#### `SMTP` class

* Represents a connection to an SMTP server.
* Methods:

  * `connect(host='localhost', port=0)` → Establish connection to SMTP server.
  * `login(user, password)` → Authenticate with SMTP server.
  * `sendmail(from_addr, to_addrs, msg)` → Send an email.
  * `quit()` → Terminate the connection.
  * `ehlo(name='')` → Identify client to server; required for extended features.
  * `starttls(keyfile=None, certfile=None, context=None)` → Upgrade to TLS.
  * `close()` → Close connection.

#### `SMTP_SSL` class

* Connects directly using SSL instead of upgrading via `starttls()`.

---

### Parameters and Defaults

```python
SMTP(
    host='',          # SMTP server hostname
    port=0,           # SMTP port (default 25 for plain, 465 for SSL, 587 for TLS)
    local_hostname=None,  # Local hostname to send to server
    timeout=None,     # Connection timeout
    source_address=None  # Source IP address tuple (host, port)
)
```

---

### Usage Scenarios

* Sending transactional emails from applications.
* Sending reports, alerts, or notifications via email.
* Sending automated newsletters or marketing campaigns.
* Sending system logs to an admin via email.

---

### Example – Send Email with TLS

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email details
sender_email = "you@example.com"
receiver_email = "recipient@example.com"
password = "yourpassword"

# Create message
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = "Test Email from Python"
body = "This is a test email sent using smtplib with TLS."
msg.attach(MIMEText(body, 'plain'))

# Connect to SMTP server
server = smtplib.SMTP('smtp.gmail.com', 587)  # Gmail TLS port
server.ehlo()
server.starttls()
server.login(sender_email, password)
server.sendmail(sender_email, receiver_email, msg.as_string())
server.quit()
```

---

### Example – Send Email with SSL

```python
import smtplib
from email.mime.text import MIMEText

msg = MIMEText("This is a test email sent using smtplib with SSL.")
msg['Subject'] = "SSL Test"
msg['From'] = "you@example.com"
msg['To'] = "recipient@example.com"

with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
    server.login("you@example.com", "yourpassword")
    server.sendmail("you@example.com", "recipient@example.com", msg.as_string())
```

---

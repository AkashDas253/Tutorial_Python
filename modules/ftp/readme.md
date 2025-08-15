## FTP (ftplib) 

### Overview

* The `ftplib` module provides tools to interact with FTP (File Transfer Protocol) servers.
* Supports plain FTP and FTPS (secure FTP over TLS).
* Implements the `FTP` class for basic operations and `FTP_TLS` for encrypted communication.
* Supports uploading, downloading, listing, renaming, deleting files, and directory management.

---

### Classes

* **`ftplib.FTP`** – Standard FTP client class.
* **`ftplib.FTP_TLS`** – Subclass of `FTP` for secure communication (FTPS).

---

### Common Methods of `FTP` Class

| Method                                                 | Description                           |
| ------------------------------------------------------ | ------------------------------------- |
| `connect(host, port=21, timeout=None)`                 | Connect to FTP server.                |
| `login(user='anonymous', passwd='', acct='')`          | Authenticate user.                    |
| `quit()`                                               | Close connection gracefully.          |
| `close()`                                              | Close without sending `QUIT`.         |
| `cwd(path)`                                            | Change working directory.             |
| `pwd()`                                                | Get current working directory.        |
| `dir([path])`                                          | List directory contents with details. |
| `nlst([path])`                                         | Get list of file names in directory.  |
| `retrbinary(cmd, callback, blocksize=8192, rest=None)` | Download binary file.                 |
| `retrlines(cmd, callback=None)`                        | Download text file line by line.      |
| `storbinary(cmd, file, blocksize=8192)`                | Upload binary file.                   |
| `storlines(cmd, file)`                                 | Upload text file.                     |
| `delete(filename)`                                     | Delete a file.                        |
| `rename(fromname, toname)`                             | Rename file.                          |
| `mkd(path)`                                            | Make directory.                       |
| `rmd(path)`                                            | Remove directory.                     |
| `set_pasv(val)`                                        | Enable/disable passive mode.          |

---

### FTP Usage Examples

#### Connect & Login

```python
from ftplib import FTP

ftp = FTP()
ftp.connect('ftp.example.com', 21)  # Default FTP port
ftp.login(user='username', passwd='password')
print("Connected:", ftp.getwelcome())  # Server welcome message
```

#### List Files in Directory

```python
ftp.cwd('/path/to/dir')
files = ftp.nlst()
print("Files:", files)
```

#### Download a Binary File

```python
with open('local_file.zip', 'wb') as f:
    ftp.retrbinary('RETR remote_file.zip', f.write)
```

#### Upload a Binary File

```python
with open('upload_file.zip', 'rb') as f:
    ftp.storbinary('STOR remote_file.zip', f)
```

#### Rename and Delete

```python
ftp.rename('old_name.txt', 'new_name.txt')
ftp.delete('unwanted_file.txt')
```

#### Create and Remove Directories

```python
ftp.mkd('new_folder')
ftp.rmd('old_folder')
```

#### Disconnect

```python
ftp.quit()
```

---

#### FTPS (Secure FTP)

```python
from ftplib import FTP_TLS

ftps = FTP_TLS()
ftps.connect('ftp.example.com', 21)
ftps.login(user='username', passwd='password')
ftps.prot_p()  # Switch to secure data connection
print(ftps.nlst())
ftps.quit()
```

---

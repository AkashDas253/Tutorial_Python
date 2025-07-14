## Deployment in Flask  

### Overview  
Flask applications can be deployed on various platforms, including:  

- **Local Servers** (Gunicorn, Waitress)  
- **Cloud Platforms** (Heroku, AWS, Google Cloud, Azure)  
- **Containerized Deployments** (Docker, Kubernetes)  

---

## 1. **Using Gunicorn (Linux/Mac) or Waitress (Windows)**  

### Install Gunicorn (for Linux/Mac)  
```sh
pip install gunicorn
```
Run Flask app:  
```sh
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```
- `-w 4`: Number of worker processes  
- `-b 0.0.0.0:8000`: Bind to port 8000  

### Install Waitress (for Windows)  
```sh
pip install waitress
```
Run Flask app:  
```sh
python -m waitress --listen=0.0.0.0:8000 app:app
```

---

## 2. **Deploying on Heroku**  
### Install Heroku CLI  
```sh
curl https://cli-assets.heroku.com/install.sh | sh  # Linux/Mac
```
OR  
```sh
choco install heroku-cli  # Windows
```

### Steps  
1. Login to Heroku  
   ```sh
   heroku login
   ```
2. Initialize Git and create a Heroku app  
   ```sh
   git init
   heroku create flask-app-name
   ```
3. Create a `Procfile` (without file extension):  
   ```
   web: gunicorn app:app
   ```
4. Add dependencies in `requirements.txt`:  
   ```sh
   pip freeze > requirements.txt
   ```
5. Deploy  
   ```sh
   git add .
   git commit -m "Initial commit"
   git push heroku master
   ```

---

## 3. **Deploying on AWS (EC2 + Nginx + Gunicorn)**  
### Steps  
1. Create an **EC2 instance**  
2. Install dependencies:  
   ```sh
   sudo apt update
   sudo apt install python3-pip nginx
   pip install flask gunicorn
   ```
3. Start Flask app with Gunicorn:  
   ```sh
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```
4. Configure **Nginx** (edit `/etc/nginx/sites-available/default`):  
   ```
   server {
       listen 80;
       server_name your_domain_or_IP;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```
5. Restart Nginx:  
   ```sh
   sudo systemctl restart nginx
   ```

---

## 4. **Deploying with Docker**  
### Install Docker  
```sh
sudo apt install docker.io  # Linux
choco install docker-cli  # Windows
```

### Create a `Dockerfile`  
```
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
```

### Build and Run  
```sh
docker build -t flask-app .
docker run -p 8000:8000 flask-app
```

---

## Summary  

| Method | Description |
|--------|------------|
| **Gunicorn/Waitress** | Local deployment for production |
| **Heroku** | Simple cloud deployment |
| **AWS (EC2 + Nginx)** | Scalable deployment with reverse proxy |
| **Docker** | Containerized deployment |

## Background Tasks in Flask  

### Overview  
Flask does not provide built-in support for background tasks, but tasks can be executed asynchronously using:  

- **Threading** (Basic concurrent execution)  
- **Celery** (Advanced task queue with workers)  
- **APScheduler** (Scheduled background jobs)  

---

## Using Threading (Simple Background Task)  
```python
import threading
import time
from flask import Flask

app = Flask(__name__)

def background_task(name):
    time.sleep(5)  # Simulating a long-running task
    print(f"Task {name} completed!")

@app.route('/start-task')
def start_task():
    thread = threading.Thread(target=background_task, args=("SampleTask",))
    thread.start()
    return {"message": "Task started"}, 202
```

---

## Using Celery (Distributed Task Queue)  

### Installation  
```sh
pip install celery redis
```

### Setting Up Celery  
```python
from flask import Flask
from celery import Celery

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
```

### Defining a Celery Task  
```python
@celery.task
def long_running_task(x, y):
    return x + y
```

### Running Celery Task from Flask  
```python
@app.route('/start-celery')
def start_celery():
    result = long_running_task.delay(10, 20)
    return {"task_id": result.id}, 202
```

### Running Celery Worker  
Start a Redis server and run the Celery worker:  
```sh
celery -A app.celery worker --loglevel=info
```

---

## Using APScheduler (Scheduled Background Jobs)  

### Installation  
```sh
pip install apscheduler
```

### Setting Up APScheduler  
```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

def scheduled_task():
    print("Scheduled task executed!")

scheduler.add_job(scheduled_task, 'interval', seconds=10)
scheduler.start()
```

---

## Summary  

| Method | Description | Use Case |
|--------|------------|----------|
| **Threading** | `threading.Thread(target=func)` | Simple tasks, no result tracking |
| **Celery** | `@celery.task` with workers | Large-scale async processing |
| **APScheduler** | `BackgroundScheduler()` | Scheduled jobs (e.g., cron jobs) |

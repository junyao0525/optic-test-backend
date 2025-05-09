
# 🚀 FastAPI Server Setup Guide

This guide explains how to configure, deploy, and manage a FastAPI application using **Gunicorn** and **systemd** on a Linux server.

---

## 📦 Prerequisites

- Python 3.8+
- Virtual environment (`venv`)
- Git
- FastAPI, Gunicorn, and Uvicorn installed
- A `.env` file (if applicable)

---

## 🔧 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

---

## 🐍 2. Create & Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ 4. Gunicorn Configuration (Optional)

Create a `gunicorn_conf.py` file:

```python
bind = "0.0.0.0:8000"
workers = 2
timeout = 120
```

---

## 🛠️ 5. Create a systemd Service File

Create the service file:

```bash
sudo nano /etc/systemd/system/fastapi.service
```

Paste the following configuration:

```ini
[Unit]
Description=FastAPI Gunicorn daemon
After=network.target

[Service]
User=azureuser
Group=www-data
WorkingDirectory=/home/azureuser/<repo-name>
ExecStart=/home/azureuser/<repo-name>/venv/bin/gunicorn -c gunicorn_conf.py main:app

[Install]
WantedBy=multi-user.target
```

> Replace `<repo-name>` with your actual project folder name.

---

## ▶️ 6. Start & Enable the Service

```bash
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable fastapi.service
sudo systemctl start fastapi.service
```

---

## 🧪 7. Verify Server Status

```bash
sudo systemctl restart fastapi.service
sudo systemctl status fastapi.service
```

For real-time logs:

```bash
journalctl -u fastapi.service -f
```

---

## 🌐 8. Access the API

Visit:

```http
http://<server-ip>:8000/docs
```

---

## 🛑 9. To Restart or Stop the Service

```bash
sudo systemctl restart fastapi.service
sudo systemctl stop fastapi.service
```

---

## ✅ Tips

- Ensure your server's port (default `8000`) is open in the firewall.
- You can run with `.env` support by including `python-dotenv` or using the `@env` loader.
- Gunicorn should point to `main:app` (adjust if your file/module name is different).

ssh -i "C:\Users\junyao\Downloads\MyLowCostVM_key.pem" azureuser@104.214.171.210

 tail -f /var/log/nginx/access.log

 tail -f /var/log/nginx/error.log
workers = 4  # Number of worker processes (adjust based on CPU cores)
worker_class = "uvicorn.workers.UvicornWorker"  # Use Uvicorn workers for ASGI
bind = "0.0.0.0:8000"  # Bind to all network interfaces
timeout = 120  # Request timeout (seconds)
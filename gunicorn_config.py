# gunicorn_conf.py

# Example config
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120

# Increase request size limits
limit_request_line = 8190  # Default is 4094
limit_request_fields = 100
max_request_line = 8190
max_request_fields = 100

# Add body size limit (10MB)
limit_request_body = 10485760  # 10MB in bytes

# Optional: Add keep-alive settings
keepalive = 5
timeout = 120
graceful_timeout = 120

# Optional: Add worker settings
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# You may need to ensure Uvicorn itself accepts large bodies (see next)

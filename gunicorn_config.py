# gunicorn_conf.py

# Example config
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120

# Optional (not directly controlling body size)
limit_request_line = 8190  # Default is 4094
limit_request_fields = 100

# You may need to ensure Uvicorn itself accepts large bodies (see next)

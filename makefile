
# local run
run:
	 uvicorn main:app --host 192.168.100.8 --port 8000 --reload

lint:
	flake8 .

docker-build:
	docker build -t my-image .

docker-run:
	docker run --rm my-image


# nginx log 
vm-access-log:
	tail -f /var/log/nginx/access.log

vm-error-log:
	tail -f /var/log/nginx/error.log



# fastapi
vm-start-fastapi:
	sudo systemctl start fastapi.service

vm-stop-fastapi:
	sudo systemctl stop fastapi.service

vm-status-fastapi:
	sudo systemctl status fastapi.service

vm-restart-fastapi:
	sudo systemctl restart fastapi.service


# nginx
vm-start-nginx:
	sudo systemctl start nginx.service

vm-stop-nginx:
	sudo systemctl stop nginx.service

vm-reload-nginx:
	sudo systemctl reload nginx.service

vm-status-nginx:
	sudo systemctl status nginx.service


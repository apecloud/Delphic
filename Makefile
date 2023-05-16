version ?= v1.0.0

build:
	docker buildx build -t apecloud/delphic:$(version) --platform linux/amd64,linux/arm64 --push -f ./compose/local/django/Dockerfile  .
	docker buildx build -t apecloud/delphic-frontend:$(version) --platform linux/amd64,linux/arm64 --push -f ./frontend/Dockerfile ./frontend

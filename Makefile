version ?= v1.0.0

build:
	docker buildx build -t apecloud/delphic:$(version) --push -f ./compose/local/django/Dockerfile  .
	docker buildx build -t apecloud/delphic-frontend:$(version) --push -f ./Dockerfile .

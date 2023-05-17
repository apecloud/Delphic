VERSION ?= v1.0.0
BUILDX_PLATFORM ?= linux/amd64,linux/arm64
BUILDX_ARGS ?= --sbom=false --provenance=false

build:
	docker buildx build -t apecloud/delphic:$(VERSION) --platform $(BUILDX_PLATFORM) $(BUILDX_ARGS) --push -f ./compose/local/django/Dockerfile  .
	docker buildx build -t apecloud/delphic-frontend:$(VERSION) --platform $(BUILDX_PLATFORM) $(BUILDX_ARGS) --push -f ./frontend/Dockerfile ./frontend

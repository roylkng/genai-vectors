delete:
	skaffold delete -p minio
	skaffold delete -p vector-store

deploy:
	skaffold deploy -p minio

dev:
	skaffold dev -p vector-store --port-forward

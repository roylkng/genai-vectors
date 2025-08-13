delete:
	skaffold delete -p minio
	skaffold delete -p qdrant
	skaffold delete -p vector-store

deploy:
	skaffold deploy -p minio
	skaffold deploy -p qdrant

dev:
	skaffold dev -p vector-store --port-forward

# Kubernetes Infrastructure (k8s)

This directory contains the Kubernetes manifests for the DANN project microservices.

## Naming Conventions
Files must follow the pattern: `<component>-<type>.yaml`
- Example: `users-app-deployment.yaml`
- Example: `posts-db-service.yaml`

## Kubernetes Standards
- **Namespace**: All resources must be deployed in the `default` namespace.
- **Labels**: Standard labels must include `app`, `tier`, and `team`.
- **Ports**: 
  - Databases must use port `5432`.
  - Application ports are defined per microservice specification.
- **Image Pull Policy**: Always use `imagePullPolicy: Never` for local Minikube development.
- **Volumes**: All database volumes must be of type `emptyDir` (ephemeral storage).

## Connection Policy
- Databases only accept connections from their corresponding application (Network Isolation).

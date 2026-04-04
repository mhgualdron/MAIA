# Grupo 12 - DANN Project

Proyecto para la asignatura **Desarrollo de Aplicaciones en la Nube (DANN)**.

## Team Members
- **m.hernandezg234@uniandes.edu.co** (Project Leader)
- **ma.quinteror1@uniandes.edu.co**
- **j.buriticar2@uniandes.edu.co**
- **jd.marinb1@uniandes.edu.co**

 
## Project Structure
```text
.
├── .github/workflows/    # CI/CD Pipelines
├── docs/                 # Documentation (Jekyll + PlantUML)
├── k8s/                  # Kubernetes Manifests
├── terraform/            # Infrastructure as Code (AWS EKS, RDS, ECR)
├── users_app/            # Microservice: Users
├── posts_app/            # Microservice: Posts
├── offers_app/           # Microservice: Offers
├── routes_app/           # Microservice: Routes
├── scores_app/           # Microservice: Scores 
├── rf003_app/            # Orchestrator: RF-003
├── rf004_app/            # Orchestrator: RF-004
├── rf005_app/            # Orchestrator: RF-005
├── credit_cards_app/     # Microservice: Credit Cards API (RF-006)
├── aws-async-poller/     # Lambda function: Async TrueNative Poller
├── config.yaml           # Global Project Configuration
└── pyproject.toml        # Root dependency management (Poetry)
```

## Tech Stack
- **Language**: Python 3.11
- **Framework**: FastAPI
- **Database**: Amazon RDS (PostgreSQL)
- **Environment**: Poetry
- **Infrastructure**: AWS (EKS, ECR, VPC) & Kubernetes
- **IaC**: Terraform
- **CI/CD**: GitHub Actions

## Installation & Deployment

### Prerequisites
- [AWS CLI](https://aws.amazon.com/cli/) configured with valid credentials.
- [Terraform](https://www.terraform.io/downloads) (v1.5+)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Docker](https://www.docker.com/get-started)
- [Poetry](https://python-poetry.org/docs/#installation)

### 1. Infrastructure (AWS)
Deploy the VPC, EKS Cluster, and RDS instance:
```bash
cd terraform
terraform init
terraform apply
```

### 2. Build and Push Deployments (ECR & Lambda)
Build the Docker images for the microservices and push them to your ECR repositories:
```bash
# Example for RF005
docker build -t <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/rf005-service:v1.0.0 ./rf005_app
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/rf005-service:v1.0.0
```

Build the AWS Lambda asynchronous poller package:
```bash
chmod +x build_lambda.sh
./build_lambda.sh
# The zip is automatically referenced by terraform apply
```

### 3. Kubernetes Deployment (EKS)
Update your kubeconfig and apply the manifests:
```bash
aws eks update-kubeconfig --name <CLUSTER_NAME> --region us-east-1
kubectl apply -f k8s/cloud/
```

## Testing
Run unit tests for any microservice:
```bash
make unittest DIR=rf005_app
```

## Documentation
The documentation is built with Jekyll and deployed to GitHub Pages.
- **Local Preview**: `cd docs && bundle exec jekyll serve`
- **Documentation Link**: [GitHub Pages](https://super-adventure-yv58poq.pages.github.io)

## CI/CD Pipelines
- `ci_evaluador_unit.yml`: Validates unit testing and coverage (>=70%).
- `ci_evaluador_entrega2.yml`: Validates the implementation of Entrega 2 requirements.
- `ci_evaluador_entrega2_k8s.yml`: Validates K8s deployment in AWS.

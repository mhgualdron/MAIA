# 🏗️ Infrastructure as Code (Terraform)

Este directorio contiene la definición completa de la infraestructura en AWS necesaria para la **Entrega 2**.

## 🛠️ Recursos Definidos
- **VPC & Networking**: Configuración de red con subnets públicas y privadas. Incluye un NAT Gateway para permitir salida a internet desde las subnets privadas.
- **Amazon EKS**: Cluster de Kubernetes gestionado (versión 1.31) con un grupo de nodos administrados.
- **Amazon RDS**: Instancia de PostgreSQL 16.12 centralizada.
- **Amazon ECR**: Repositorios para almacenar las imágenes Docker de todos los microservicios.

## 🚀 Despliegue de Infraestructura

### Requisitos Previos
- AWS CLI configurado con las credenciales de **AWS Academy Learner Lab**.
- Terraform instalado.

### Pasos
1. **Inicializar**:
   ```bash
   terraform init
   ```
2. **Planificar**:
   ```bash
   terraform plan
   ```
3. **Aplicar**:
   ```bash
   terraform apply -auto-approve
   ```

## 🔐 Seguridad y Acceso
- **Base de Datos**: El RDS solo es accesible desde dentro de la VPC por los pods del cluster EKS.
- **EKS**: Los nodos utilizan el `LabRole` pre-existente en AWS Academy para cumplir con las restricciones de IAM de la plataforma.

## 🔄 Salidas (Outputs)
Al finalizar el despliegue, Terraform proporcionará los siguientes datos críticos:
- `eks_cluster_name`: Nombre del cluster para conectar `kubectl`.
- `rds_endpoint`: URL de conexión a la base de datos central.
- `ecr_repository_urls`: Lista de URLs para subir las imágenes Docker.

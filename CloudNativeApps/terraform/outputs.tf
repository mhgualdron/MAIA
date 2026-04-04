output "ecr_registry" {
  description = "ECR registry base URL (use as prefix for all image names)"
  value       = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com"
}

output "ecr_repository_urls" {
  description = "Full URL for each ECR repository"
  value       = { for k, v in aws_ecr_repository.repos : k => v.repository_url }
}

# Data source to get current AWS account ID
data "aws_caller_identity" "current" {}

# NOTE: RDS and EKS outputs will be added when rds.tf and eks.tf are created.

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint (host:port)"
  value       = aws_db_instance.dann_postgres.endpoint
}

output "rds_host" {
  description = "RDS PostgreSQL hostname"
  value       = aws_db_instance.dann_postgres.address
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "eks_cluster_endpoint" {
  description = "EKS API server endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "sqs_queue_url" {
  description = "SQS Queue URL for Credit Cards Poller"
  value       = aws_sqs_queue.credit_cards_queue.url
}

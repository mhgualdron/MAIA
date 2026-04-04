variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "dann"
}

variable "team" {
  description = "Team identifier for tagging"
  type        = string
  default     = "grupo12"
}

variable "cluster_name" {
  description = "EKS cluster name (must match config.yaml cluster_name)"
  type        = string
  default     = "dann-eks-cluster"
}

variable "db_name" {
  description = "RDS master database name"
  type        = string
  default     = "dann_db"
}

variable "db_username" {
  description = "RDS master username"
  type        = string
  default     = "postgres"
  sensitive   = true
}

variable "db_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "node_instance_type" {
  description = "EC2 instance type for EKS worker nodes"
  type        = string
  default     = "t3.medium"
}

variable "node_desired_size" {
  description = "Desired number of EKS worker nodes"
  type        = number
  default     = 2
}

variable "node_min_size" {
  description = "Minimum number of EKS worker nodes"
  type        = number
  default     = 2
}

variable "node_max_size" {
  description = "Maximum number of EKS worker nodes"
  type        = number
  default     = 4
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnets" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnets" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24"]
}


variable "url" {
  description = "The base URL/DNS for the application (Ingress ELB)"
  type        = string
  default     = "aa5f84d48ce6c452d9d05486b7d9b503-7cee93fd98292a44.elb.us-east-1.amazonaws.com"

}

variable "secret_token" {
  description = "Secret token for TrueNative verification sharing"
  type        = string
  default     = "random_secret_token"
}

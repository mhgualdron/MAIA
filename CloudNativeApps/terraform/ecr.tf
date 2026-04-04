locals {
  ecr_repositories = [
    # Delivery 1 apps
    "users-app",
    "posts-app",
    "offers-app",
    "routes-app",
    # Delivery 2 - new services (Team Lead)
    "scores-app",
    # Delivery 2 - RF orchestrators (teammates)
    "rf003-app",
    "rf004-app",
    "rf005-app",
    "rf003-service",
    "rf004-service",
    "rf005-service",
    # Delivery 3 - RF-006
    "credit-cards-app",
  ]
}




resource "aws_ecr_repository" "repos" {
  for_each = toset(local.ecr_repositories)

  name                 = each.value
  image_tag_mutability = "MUTABLE"
  force_delete         = true


  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project = var.project
    Team    = var.team
  }
}

# Lifecycle policy: keep only the last 5 images per repo to save storage costs
resource "aws_ecr_lifecycle_policy" "repos" {
  for_each   = aws_ecr_repository.repos
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

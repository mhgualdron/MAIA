resource "aws_db_instance" "dann_postgres" {
  identifier           = "${var.project}-db"
  allocated_storage    = 20
  storage_type        = "gp2"
  engine              = "postgres"
  engine_version      = "16.12"
  instance_class      = "db.t3.micro"
  db_name             = var.db_name
  username            = var.db_username
  password            = var.db_password
  parameter_group_name = "default.postgres16"

  skip_final_snapshot  = true
  publicly_accessible = false

  db_subnet_group_name   = aws_db_subnet_group.db_subnet_group.name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]

  tags = {
    Project = var.project
    Team    = var.team
  }
}

resource "aws_db_subnet_group" "db_subnet_group" {
  name       = "${var.project}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Project = var.project
    Team    = var.team
  }
}

resource "aws_security_group" "rds_sg" {
  name        = "${var.project}-rds-sg"
  description = "Allow inbound PostgreSQL traffic from VPC"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "PostgreSQL from VPC"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project = var.project
    Team    = var.team
  }
}

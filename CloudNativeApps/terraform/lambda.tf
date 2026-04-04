# We cannot create IAM Roles in AWS Academy. We must use the pre-existing LabRole.
data "aws_iam_role" "lab_role" {
  name = "LabRole"
}

resource "aws_lambda_function" "credit_cards_poller" {
  filename         = "../aws-async-poller/deployment_package.zip"
  source_code_hash = filebase64sha256("../aws-async-poller/deployment_package.zip")
  function_name    = "${var.project}-async-poller"
  role             = data.aws_iam_role.lab_role.arn
  handler          = "lambda_function.lambda_handler"
  runtime          = "python3.11"
  timeout          = 45
  memory_size      = 256

  vpc_config {
    subnet_ids         = module.vpc.private_subnets
    security_group_ids = [aws_security_group.rds_sg.id]
  }

  environment {
    variables = {
      DB_URI            = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.dann_postgres.endpoint}/${var.db_name}"
      TRUENATIVE_HOST   = "http://aa5f84d48ce6c452d9d05486b7d9b503-7cee93fd98292a44.elb.us-east-1.amazonaws.com"
      TRUENATIVE_SECRET = "random_secret_token"
      SES_SENDER_EMAIL  = "ma.quinteror1@uniandes.edu.co"
    }
  }

  tags = {
    Project = var.project
    Team    = var.team
  }
}

resource "aws_lambda_event_source_mapping" "sqs_mapping" {
  event_source_arn = aws_sqs_queue.credit_cards_queue.arn
  function_name    = aws_lambda_function.credit_cards_poller.arn
  batch_size       = 10
}

# SNS Topic for User Events
# SNS Topic for User Events
resource "aws_sns_topic" "user_events" {
  name = "${var.project}-${var.team}-user-events"
  tags = {
    Project = var.project
    Team    = var.team
  }
}

locals {
  lab_role_arn = data.aws_iam_role.lab_role.arn
}


# --- Verification Trigger Lambda ---

data "archive_file" "verification_trigger_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../serverless/verification_trigger"
  output_path = "${path.module}/../serverless/verification_trigger.zip"
}

resource "aws_lambda_function" "verification_trigger" {
  filename      = data.archive_file.verification_trigger_zip.output_path
  function_name = "${var.project}-${var.team}-verification-trigger"
  role          = local.lab_role_arn

  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.11"

  environment {
    variables = {
      TRUENATIVE_URL    = "http://${var.url}/native/verify"
      CALLBACK_BASE_URL = "http://${var.url}/users/verify/callback"
      SECRET_TOKEN      = var.secret_token

    }
  }

  source_code_hash = data.archive_file.verification_trigger_zip.output_base64sha256
}

resource "aws_lambda_permission" "sns_trigger_verification" {
  statement_id  = "AllowExecutionFromSNS"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.verification_trigger.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.user_events.arn
}

resource "aws_sns_topic_subscription" "verification_trigger_sub" {
  topic_arn = aws_sns_topic.user_events.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.verification_trigger.arn
  
  filter_policy = jsonencode({
    event_type = ["USER_CREATED"]
  })
}

# --- Notification Service Lambda ---

data "archive_file" "notification_service_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../serverless/notification_service"
  output_path = "${path.module}/../serverless/notification_service.zip"
}

resource "aws_lambda_function" "notification_service" {
  filename      = data.archive_file.notification_service_zip.output_path
  function_name = "${var.project}-${var.team}-notification-service"
  role          = local.lab_role_arn

  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.11"

  # No environment variables needed for direct code credentials

  source_code_hash = data.archive_file.notification_service_zip.output_base64sha256
}

resource "aws_lambda_permission" "sns_trigger_notification" {
  statement_id  = "AllowExecutionFromSNS"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.notification_service.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.user_events.arn
}

resource "aws_sns_topic_subscription" "notification_service_sub" {
  topic_arn = aws_sns_topic.user_events.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.notification_service.arn

  filter_policy = jsonencode({
    event_type = ["VERIFICATION_COMPLETED"]
  })
}

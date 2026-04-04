resource "aws_sqs_queue" "credit_cards_queue" {
  name                      = "${var.project}-credit-cards-queue"
  delay_seconds             = 0
  max_message_size          = 262144
  message_retention_seconds = 86400
  receive_wait_time_seconds  = 10
  visibility_timeout_seconds = 60

  tags = {
    Project = var.project
    Team    = var.team
  }
}

import boto3
import json
from src.core.config import settings

def publish_card_event(card_id: str, ruv: str, user_id: str, email: str, created_at: str):
    if not settings.SQS_QUEUE_URL.startswith("http"):
        print("SQS no configurado, omitiendo evento.")
        return

    try:
        sqs = boto3.client('sqs', region_name=settings.AWS_REGION)
        payload = {
            "cardId": card_id,
            "ruv": ruv,
            "userId": user_id,
            "userEmail": email,
            "createdAt": created_at
        }
        print(f"Publishing event for card {card_id} with RUV {ruv}")
        sqs.send_message(
            QueueUrl=settings.SQS_QUEUE_URL,
            MessageBody=json.dumps(payload)
        )
    except Exception as e:
        print(f"Error publishing SQS event: {e}")

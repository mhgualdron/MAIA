import boto3
import os
import json
import logging

logger = logging.getLogger(__name__)

class SNSService:
    def __init__(self):
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.topic_arn = os.getenv('USER_EVENTS_TOPIC_ARN')
        self._sns = None

    def _get_sns_client(self):
        if self._sns:
            return self._sns
        
        if not self.topic_arn:
            logger.warning("USER_EVENTS_TOPIC_ARN not set")
            return None

        try:
            self._sns = boto3.client('sns', region_name=self.region)
            return self._sns
        except Exception as e:
            logger.error(f"Failed to initialize SNS client: {e}")
            return None

    def publish_user_created(self, user_id: str, email: str, dni: str = None, full_name: str = None, phone: str = None):
        sns = self._get_sns_client()
        if not sns or not self.topic_arn:
            logger.warning("SNS not configured, skipping USER_CREATED event")
            return

            
        message = {
            "event": "USER_CREATED",
            "data": {
                "userIdentifier": user_id,
                "email": email,
                "dni": dni,
                "fullName": full_name,
                "phone": phone
            }
        }
        
        try:
            sns.publish(
                TopicArn=self.topic_arn,
                Message=json.dumps(message),
                MessageAttributes={
                    'event_type': {
                        'DataType': 'String',
                        'StringValue': 'USER_CREATED'
                    }
                }
            )
            logger.info(f"Published USER_CREATED event for user {user_id}")
            print(f"DEBUG: Published USER_CREATED event for user {user_id}")
        except Exception as e:
            logger.error(f"Error publishing USER_CREATED event: {e}")
            print(f"DEBUG: Error publishing USER_CREATED event: {e}")


    def publish_verification_completed(self, user_id: str, email: str, status: str, ruv: str):
        sns = self._get_sns_client()
        if not sns or not self.topic_arn:
            logger.warning("SNS not configured, skipping VERIFICATION_COMPLETED event")
            return


        message = {
            "event": "VERIFICATION_COMPLETED",
            "data": {
                "userIdentifier": user_id,
                "email": email,
                "status": status,
                "ruv": ruv
            }
        }

        try:
            sns.publish(
                TopicArn=self.topic_arn,
                Message=json.dumps(message),
                MessageAttributes={
                    'event_type': {
                        'DataType': 'String',
                        'StringValue': 'VERIFICATION_COMPLETED'
                    }
                }
            )
            logger.info(f"Published VERIFICATION_COMPLETED event for user {user_id}")
            print(f"DEBUG: Published VERIFICATION_COMPLETED event for user {user_id}")
        except Exception as e:
            logger.error(f"Error publishing VERIFICATION_COMPLETED event: {e}")
            print(f"DEBUG: Error publishing VERIFICATION_COMPLETED event: {e}")


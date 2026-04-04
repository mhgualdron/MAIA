import json
import os
import urllib3

http = urllib3.PoolManager()

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")
    
    # 1. Parse SNS Message
    try:
        sns_message = json.loads(event['Records'][0]['Sns']['Message'])
        event_type = sns_message.get('event')
        data = sns_message.get('data')
    except Exception as e:
        print(f"Error parsing SNS message: {e}")
        return {"statusCode": 400, "body": "Invalid SNS message"}

    if event_type != "USER_CREATED":
        print(f"Skipping event type: {event_type}")
        return {"statusCode": 200, "body": "Skipped"}

    # 2. Get configuration
    truenative_url = os.getenv('TRUENATIVE_URL') # e.g. http://<host>/native/verify
    secret_token = os.getenv('SECRET_TOKEN')
    callback_base_url = os.getenv('CALLBACK_BASE_URL') # e.g. http://<host>/users/verify/callback

    if not truenative_url:
        print("Configuration missing: TRUENATIVE_URL")
        return {"statusCode": 500, "body": "Configuration missing"}

    # 3. Call TrueNative
    verification_request = {
        "user": {
            "email": data.get('email'),
            "dni": data.get('dni'),
            "fullName": data.get('fullName'),
            "phone": data.get('phone')
        },
        "transactionIdentifier": f"verify_{data.get('userIdentifier')}_{int(context.aws_request_id[-5:], 16)}",
        "userIdentifier": data.get('userIdentifier'),
        "userWebhook": callback_base_url
    }

    try:
        resp = http.request(
            'POST',
            truenative_url,
            body=json.dumps(verification_request),
            headers={
                'Content-Type': 'application/json',
                'Authorization': secret_token
            }
        )

        print(f"TrueNative response: {resp.status} - {resp.data.decode('utf-8')}")
        return {
            "statusCode": resp.status,
            "body": resp.data.decode('utf-8')
        }
    except Exception as e:
        print(f"Error calling TrueNative: {e}")
        return {"statusCode": 500, "body": str(e)}

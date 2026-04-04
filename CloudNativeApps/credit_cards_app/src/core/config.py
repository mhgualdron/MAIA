import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_URI = os.getenv("DB_URI", "postgresql://admin:admin@localhost:5432/credit_cards_db")
    TRUENATIVE_HOST = os.getenv("TRUENATIVE_HOST", "http://truenative.default.svc.cluster.local")
    TRUENATIVE_SECRET = os.getenv("TRUENATIVE_SECRET", "")
    USERS_APP_HOST = os.getenv("USERS_APP_HOST", "http://users-app-svc:80")
    SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

settings = Config()

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/routes_db")
    APP_NAME: str = "Routes App"
    APP_VERSION: str = "0.1.0"
    
    class Config:
        env_file = ".env"

settings = Settings()

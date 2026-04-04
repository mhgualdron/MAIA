from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    users_service_url: str = "http://users-app-service:80"
    offers_service_url: str = "http://offers-app-service:80"
    posts_service_url: str = "http://posts-app-service:80"
    routes_service_url: str = "http://routes-app-service:80"
    scores_service_url: str = "http://scores-app-service:80"

    class Config:
        env_file = ".env"

settings = Settings()

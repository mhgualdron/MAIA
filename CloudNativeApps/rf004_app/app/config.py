from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Basado en tu comando 'kubectl get svc'
    # Users y Offers están en el 3000. Posts, Routes y Score están en el 8000.
    
    users_service_url: str = "http://users-app-service:3000"
    offers_service_url: str = "http://offers-app-service:3000"
    
    posts_service_url: str = "http://posts-app-service:8000"
    routes_service_url: str = "http://routes-app-service:8000"
    score_service_url: str = "http://score-app-service:8000"

    class Config:
        env_file = ".env"

settings = Settings()
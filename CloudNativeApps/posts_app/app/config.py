from pydantic_settings import BaseSettings
from pydantic import ConfigDict

# Configuración de la aplicación y de la base de datos
class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/posts_db"  # URL de conexión a la base de datos
    app_name: str = "Posts App"  # Nombre de la aplicación
    app_version: str = "1.0.0"   # Versión de la aplicación

    # Configuración de Pydantic para leer variables desde un archivo .env
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

# Instancia global de configuración
settings = Settings()
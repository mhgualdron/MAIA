# Imagen base
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Instalar Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Configurar Poetry para no usar virtualenvs
RUN poetry config virtualenvs.create false

# Copiar archivos de dependencias
COPY pyproject.toml poetry.lock* ./

# Instalar dependencias
RUN poetry install --no-interaction --no-ansi --no-root --only main

# Copiar código fuente
COPY . .

# Exponer puerto
EXPOSE 8000

# Ejecutar la aplicación esperando a la base de datos
CMD ["sh", "-c", "until pg_isready -h posts-db-service -p 5432 -U postgres; do echo waiting for database; sleep 2; done; uvicorn app.main:app --host 0.0.0.0 --port 8000"]
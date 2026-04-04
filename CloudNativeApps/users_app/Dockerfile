# Multistage build
# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install Poetry 2.x
RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0"

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies without dev dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-root --no-interaction

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install postgresql-client for pg_isready
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY ./app ./app

# Expose port
EXPOSE 3000

# Run the application
CMD ["sh", "-c", "until pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; do echo waiting for database; sleep 2; done; uvicorn app.main:app --host 0.0.0.0 --port 3000"]

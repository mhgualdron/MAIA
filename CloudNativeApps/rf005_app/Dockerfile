FROM python:3.11-slim

WORKDIR /app

# Install Poetry without curl (using pip instead)
RUN pip install --no-cache-dir poetry

# Configure poetry to not use virtualenvs inside container
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root --only main

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

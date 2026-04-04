# Posts App

Microservice responsible for managing posts (publicaciones).

## Tech Stack
- Python 3.11
- FastAPI
- PostgreSQL (async + SQLAlchemy)
- Poetry

## Setup

### Local
```bash
poetry install
poetry run uvicorn app.main:app --reload


### Docker
```bash
docker build -t posts-app .
docker run -p 8000:8000 routes-app
```

## API
POST /posts: Create a post
GET /posts: List posts (with optional filters)
GET /posts/{id}: Get post details by ID
DELETE /posts/{id}: Delete a post by ID
GET /posts/ping: Healthcheck endpoint
POST /posts/reset: Reset all posts (delete all entries)

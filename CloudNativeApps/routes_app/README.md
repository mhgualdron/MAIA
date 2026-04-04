# Routes App

Microservice responsible for managing flight routes (trayectos).

## Tech Stack
- Python 3.11
- FastAPI
- PostgreSQL (AsyncPG + SQLAlchemy)
- Poetry

## Setup

### Local
```bash
poetry install
poetry run uvicorn app.main:app --reload
```

### Docker
```bash
docker build -t routes-app .
docker run -p 8000:8000 routes-app
```

## API
- `POST /routes`: Create a route
- `GET /routes`: List routes
- `GET /routes/{id}`: Get route details
- `DELETE /routes/{id}`: Delete route

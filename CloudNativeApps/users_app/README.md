# Users Microservice

Microservicio encargado de la gestión de usuarios, autenticación y seguridad para el proyecto DANN.

## Tecnologías
- **Language**: Python 3.11
- **Framework**: FastAPI
- **ORM**: SQLAlchemy
- **DB**: PostgreSQL 15
- **Package Manager**: Poetry
- **Testing**: Pytest

## Estructura
Siguiendo Clean Architecture:
```
users_app/
├── app/
│   ├── repositories/ # Acceso a datos
│   ├── services/     # Lógica de negocio
│   ├── routers/      # Endpoints HTTP
│   ├── models.py     # Entidades BD
│   └── schemas.py    # DTOs
├── tests/            # Unit tests (>70% coverage)
├── Dockerfile        # Multi-stage build
└── pyproject.toml    # Dependencias
```

## Setup Local
1. Instalar Poetry 2.x
2. `cd users_app`
3. `poetry install` (instala deps)
4. `poetry run pytest` (ejecuta tests)

## Docker Build
```bash
docker build -t users-app:v1.0.0 -f users_app/Dockerfile users_app
```

## Endpoints Principales
- `POST /users`: Registro de usuario
- `POST /users/auth`: Login (retorna token UUID)
- `GET /users/me`: Perfil del usuario autenticado
- `PATCH /users/{id}`: Actualizar datos

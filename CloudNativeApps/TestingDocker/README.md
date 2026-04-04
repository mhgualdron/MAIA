# ☁️ CloudNativeApps — Testing Docker

Ejercicios introductorios de **contenedorización con Docker** del módulo Desarrollo de Aplicaciones Nativas en la Nube (DANN).

## Objetivo

Aprender los conceptos básicos de Docker: imágenes, contenedores, Dockerfile, redes y volúmenes, antes de pasar al desarrollo de microservicios.

## Tópicos cubiertos

- Escribir y construir Dockerfiles
- Comandos básicos: `docker build`, `docker run`, `docker ps`, `docker exec`
- Redes Docker y comunicación entre contenedores
- Volúmenes para persistencia
- Docker Compose para multi-contenedor

## Cómo usar

```bash
# Construir una imagen
docker build -t my-image .

# Correr un contenedor
docker run -p 8080:8080 my-image

# Ver contenedores activos
docker ps
```

## Referencia

- [Docker Docs](https://docs.docker.com/)
- [Play with Docker](https://labs.play-with-docker.com/)

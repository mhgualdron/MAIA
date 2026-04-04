# RAG — Retrieval-Augmented Generation

Sistema de **RAG** (Retrieval-Augmented Generation) implementado como API REST con FastAPI. Permite hacer preguntas sobre documentos privados usando un LLM aumentado con recuperación de contexto.

## Arquitectura

```
Query → Embedding → Vector DB Search → Context → LLM → Response
```

## Estructura

```
RAG/
├── main.py          # FastAPI application entrypoint
├── routers/         # API route handlers
├── utils/           # Embedding, retrieval, LLM utilities
├── database/        # Vector DB client and indexing
└── datasets/        # Documents to index
```

## 🚀 Cómo ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Levantar la API
uvicorn main:app --reload --port 8000

# Endpoints disponibles
# POST /query   - Hacer una pregunta
# POST /ingest  - Indexar un documento
# GET  /health  - Health check
```

## Tech Stack

- **Framework**: FastAPI
- **Embeddings**: sentence-transformers / OpenAI embeddings
- **Vector DB**: ChromaDB / Pinecone
- **LLM**: OpenAI GPT / Anthropic Claude / modelos locales

## Referencia

- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB](https://www.trychroma.com/)

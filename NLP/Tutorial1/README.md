# NLP — Tutorial 1: Modelos de Lenguaje

## Descripción

Introducción práctica a los modelos de lenguaje: desde modelos N-gram y estadísticos hasta redes neuronales recurrentes y Transformers aplicados a tareas de generación y comprensión de texto.

## Contenido

| Notebook | Descripción |
|----------|-------------|
| [`modelos_de_lenguaje.ipynb`](./modelos_de_lenguaje.ipynb) | Tutorial completo de modelos de lenguaje: perplexity, N-gram, modelos neuronales, evaluación |

## Tópicos cubiertos

- Modelos de lenguaje estadísticos (bigramas, trigramas)
- Perplexity como métrica de evaluación
- Redes neuronales recurrentes para modelado de lenguaje
- Introducción a Transformers en NLP
- Generación de texto y muestreo (greedy, top-k, nucleus)

## Cómo ejecutar

```bash
# Colab (recomendado)
# Runtime → T4 GPU

# Local
pip install torch transformers
jupyter notebook modelos_de_lenguaje.ipynb
```

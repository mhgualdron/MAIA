# 🔬 Developments — Experimentos con LLMs

Área de experimentación con modelos de lenguaje de gran escala (LLMs). Contiene proyectos de investigación aplicada en Fine-Tuning y RAG (Retrieval-Augmented Generation).

---

## 📚 Contenido

| Módulo | Descripción | Estado |
|--------|-------------|--------|
| [FineTunning/](./FineTunning/) | Fine-tuning de modelos pre-entrenados para tareas específicas | 🚧 WIP |
| [RAG/](./RAG/) | Sistema RAG: FastAPI + vector DB + LLM | ✅ Funcional |

---

## 🎯 Objetivos

- Explorar técnicas de adaptación eficiente de LLMs (LoRA, QLoRA, PEFT)
- Implementar pipelines de RAG con bases de datos vectoriales
- Comparar calidad de respuestas pre/post fine-tuning
- Desplegar LLMs como APIs REST con FastAPI

---

## 🚀 Configuración del entorno

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🔗 Referencias

- [HuggingFace PEFT (LoRA)](https://huggingface.co/docs/peft)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

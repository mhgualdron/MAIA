# 🤖 Deep Learning

Módulo de **Deep Learning** del programa MAIA. Cubre las arquitecturas y técnicas fundamentales del aprendizaje profundo, desde redes convolucionales hasta transformers y modelos generativos.

---

## 📚 Contenido

| Recurso | Tipo | Descripción |
|---------|------|-------------|
| [Laboratorios/](./Laboratorios/) | Práctica guiada | CNNs, RNNs, Transformers, ViT |
| [Talleres/](./Talleres/) | Tutoriales de framework | PyTorch, TensorFlow, VAE |
| [Microproyecto1/](./Microproyecto1/) | Proyecto evaluado | Clasificación de imágenes con CNN |
| [Microproyecto2/](./Microproyecto2/) | Proyecto evaluado | Modelos secuenciales (RNN/LSTM) |
| [Microproyecto4/](./Microproyecto4/) | Proyecto evaluado | Proyecto con ViT / arquitectura avanzada |
| [AudioAPP/](./AudioAPP/) | Aplicación | API FastAPI para clasificación de audio |

---

## 🎯 Objetivos de aprendizaje

- Implementar y entrenar CNNs para clasificación y detección de objetos
- Comprender y aplicar RNNs y LSTMs para datos secuenciales
- Utilizar la arquitectura Transformer para tareas de lenguaje y visión
- Aplicar Vision Transformers (ViT) en tareas de visión por computadora
- Explorar modelos generativos (VAE)
- Desplegar modelos de DL como microservicios (FastAPI)

---

## 🚀 Configuración del entorno

```bash
# Instalar dependencias
pip install -r requirements.txt

# Alternativa: crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.\.venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

**Dependencias principales:**
```
torch torchvision tensorflow keras transformers
numpy pandas matplotlib scikit-learn
```

> 💡 La mayoría de los notebooks están optimizados para **Google Colab** con GPU T4/A100.

---

## 📂 Laboratorios

### `Laboratorios/`

| Notebook | Tópico |
|----------|--------|
| `laboratorioCNN.ipynb` | Redes Neuronales Convolucionales — fundamentos |
| `laboratorioCNN2-Copy1.ipynb` | CNNs — continuación y transfer learning |
| `LaboratorioRNN.ipynb` | Redes Neuronales Recurrentes |
| `LaboratorioRNN2-Copy1.ipynb` | RNNs — secuencias y predicción |
| `LaboratorioTransformersDeLenguajeV3.ipynb` | Transformers para NLP |
| `LaboratorioViTParte1.ipynb` | Vision Transformer — Parte 1 |
| `LaboratorioViTParte2.ipynb` | Vision Transformer — Parte 2 |

### `Talleres/`

| Notebook | Tópico |
|----------|--------|
| `TallerRedesNeuronales1.ipynb` | Redes neuronales densas desde cero |
| `TallerRedesNeuronales2.ipynb` | Redes neuronales — optimización y regularización |
| `TutorialPytorch.ipynb` | Introducción completa a PyTorch |
| `Tutorial_Tensorflow.ipynb` | Introducción a TensorFlow/Keras |
| `TallerVAEP2.ipynb` | Variational Autoencoders |

---

## 🔗 Referencias

- [PyTorch Docs](https://pytorch.org/docs/)
- [TensorFlow Docs](https://www.tensorflow.org/api_docs)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [An Image is Worth 16x16 Words — ViT Paper](https://arxiv.org/abs/2010.11929)
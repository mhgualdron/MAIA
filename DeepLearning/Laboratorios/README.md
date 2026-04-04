# 🔬 Laboratorios — Deep Learning

Notebooks de práctica guiada para los conceptos fundamentales de Deep Learning.

## Contenido

| Notebook | Tópico | Frameworks |
|----------|--------|------------|
| [`laboratorioCNN.ipynb`](./laboratorioCNN.ipynb) | CNNs: convolución, pooling, arquitecturas LeNet/VGG | PyTorch |
| [`laboratorioCNN2-Copy1.ipynb`](./laboratorioCNN2-Copy1.ipynb) | CNNs: transfer learning y fine-tuning | PyTorch |
| [`LaboratorioRNN.ipynb`](./LaboratorioRNN.ipynb) | RNNs: células recurrentes, BPTT, problemas de gradiente | PyTorch |
| [`LaboratorioRNN2-Copy1.ipynb`](./LaboratorioRNN2-Copy1.ipynb) | LSTMs y GRUs para predicción de secuencias | PyTorch |
| [`LaboratorioTransformersDeLenguajeV3.ipynb`](./LaboratorioTransformersDeLenguajeV3.ipynb) | Arquitectura Transformer completa (atención, encoders, decoders) | PyTorch |
| [`LaboratorioViTParte1.ipynb`](./LaboratorioViTParte1.ipynb) | Vision Transformer (ViT): patchification y self-attention | PyTorch |
| [`LaboratorioViTParte2.ipynb`](./LaboratorioViTParte2.ipynb) | ViT: fine-tuning en datasets de imágenes | PyTorch |

## Cómo ejecutar

Los laboratorios están diseñados para **Google Colab** (recomendado con GPU):

1. Abre el notebook en Colab: `File → Open notebook → GitHub`
2. Activa GPU: `Runtime → Change runtime type → T4 GPU`
3. Ejecuta las celdas en orden

O en local:
```bash
jupyter notebook laboratorioCNN.ipynb
```

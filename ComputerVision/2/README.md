# Computer Vision — Parte 2: NeRF

Notebooks de la segunda parte del módulo: representaciones 3D neurales.

## Contenido

| Notebook | Descripción |
|----------|-------------|
| [`NeRF_tutorial.ipynb`](./NeRF_tutorial.ipynb) | Tutorial completo de Neural Radiance Fields (NeRF): síntesis de vistas nuevas, volumen rendering, entrenamiento desde múltiples imágenes |

## Sobre NeRF

NeRF representa una escena 3D como una función implícita (MLP) que mapea posición + dirección de vista → color + densidad. Permite sintetizar vistas fotorrealistas de la escena desde ángulos no observados durante el entrenamiento.

## Cómo ejecutar

> ⚠️ Este notebook requiere **GPU** (mínimo 8GB VRAM). Se recomienda Google Colab A100.

```bash
# Google Colab:
# Runtime → Change runtime type → A100 GPU
```

## Referencia

- [NeRF Paper (Mildenhall et al. 2020)](https://arxiv.org/abs/2003.08934)
- [NeRF Studio](https://docs.nerf.studio/)

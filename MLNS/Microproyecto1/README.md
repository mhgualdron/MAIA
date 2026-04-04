# Microproyecto 1 — Clasificación de Textos ODS

## Descripción

Clasificación multi-label de textos en los 17 **Objetivos de Desarrollo Sostenible** (ODS) de las Naciones Unidas. El modelo recibe un texto y predice a cuáles ODS está relacionado.

## Dataset

| Archivo | Descripción |
|---------|-------------|
| [`Train_textosODS.xlsx`](./Train_textosODS.xlsx) | Dataset de entrenamiento con textos etiquetados por ODS |
| [`osd.csv`](./osd.csv) | Dataset ampliado de textos ODS (~21MB) |

## Contenido del proyecto

| Archivo | Descripción |
|---------|-------------|
| [`microproyecto.ipynb`](./microproyecto.ipynb) | Notebook principal: EDA, preprocesamiento, modelado, evaluación |

## Pipeline

1. **EDA**: distribución de etiquetas, longitud de textos, co-ocurrencias
2. **Preprocesamiento**: limpieza, tokenización, vectorización (TF-IDF / embeddings)
3. **Modelado**: clasificadores multi-label (OneVsRest, Chain, redes neuronales)
4. **Evaluación**: Hamming Loss, Micro/Macro F1, Precision@K

## Cómo ejecutar

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl
jupyter notebook microproyecto.ipynb
```

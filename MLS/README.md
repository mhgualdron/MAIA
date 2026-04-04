# 🏥 MLS — Machine Learning en Salud

Módulo de **Machine Learning aplicado a la Salud** del programa MAIA. Explora el uso de técnicas de ML para resolver problemas clínicos y biomédicos, con énfasis en interpretabilidad, manejo de datos desbalanceados y validación clínica.

---

## 📚 Contenido

| Archivo | Descripción |
|---------|-------------|
| [`ProyectoFinal.ipynb`](./ProyectoFinal.ipynb) | Proyecto final: modelo predictivo clínico completo |
| [`ProyectoFinal.html`](./ProyectoFinal.html) | Render HTML del notebook (para visualización sin Jupyter) |
| [`data_train.csv`](./data_train.csv) | Dataset de entrenamiento (~1.5M registros) |

---

## 🎯 Objetivos de aprendizaje

- Preprocesar données clínicas: imputación, encoding, normalización
- Manejar desbalance de clases (SMOTE, class weights, threshold tuning)
- Implementar y comparar modelos supervisados (LR, RF, XGBoost, redes neuronales)
- Evaluar con métricas clínicas: AUC-ROC, sensibilidad, especificidad, PPV
- Interpretar modelos con SHAP y LIME para uso clínico

---

## 🚀 Cómo ejecutar

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap
jupyter notebook ProyectoFinal.ipynb
```

> ⚠️ El dataset `data_train.csv` (~1.5GB) puede tardar en cargarse. Se recomienda al menos 8GB de RAM.

> 💡 Para visualización rápida sin ejecutar, abrir `ProyectoFinal.html` en el navegador.

---

## 🔗 Referencias

- [SHAP: Explainable AI](https://shap.readthedocs.io/)
- [imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)
- [Scikit-learn Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)

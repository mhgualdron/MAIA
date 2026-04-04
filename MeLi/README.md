# 🛒 MeLi — Challenge Técnico MercadoLibre

Challenge técnico de MercadoLibre completado como parte del proceso de selección / proyecto académico. Implementación de una solución de clasificación / búsqueda de productos usando Machine Learning.

---

## 📚 Estructura

```
MeLi/
├── src/          # Código fuente Python
├── Golang/       # Implementación / integración en Go
└── tests/        # Suite de tests
```

---

## 🎯 Descripción del Challenge

> Sistema de ML para clasificación automática de productos del marketplace de MercadoLibre.

Funcionalidades implementadas:
- Pipeline de preprocesamiento de datos de productos
- Modelo de clasificación de categorías
- API de inferencia para clasificación en tiempo real
- Tests unitarios e integración

---

## 🚀 Cómo ejecutar

### Python

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar tests
pytest tests/

# Levantar servicio
python src/main.py
```

### Go

```bash
cd Golang/
go mod tidy
go run main.go
```

---

## Tech Stack

- **ML**: Python · scikit-learn · pandas
- **Backend**: Python (FastAPI / Flask) + Go
- **Testing**: pytest

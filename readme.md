# API REST para Predicción de Cáncer de Mama con Docker

## Descripción del Proyecto

Esta aplicación implementa una **API REST contenerizada** usando Flask que permite consumir un modelo de Machine Learning entrenado para clasificar tumores de mama como benignos o malignos. El modelo utiliza el dataset de Breast Cancer de Wisconsin y está basado en un **RandomForestClassifier** con preprocesamiento **StandardScaler**.

## Estructura del Proyecto

```
proyecto/
├── train_model.py      # Script para entrenar el modelo (versión mejorada)
├── app.py              # API REST con Flask
├── predict_model.py    # Clase para cargar modelo y realizar predicciones
├── test_api.py         # Script de pruebas automatizadas
├── requirements.txt    # Dependencias del proyecto
├── Dockerfile          # Configuración para contenerización
├── modelo.pkl          # Modelo entrenado (generado)
├── scaler.pkl          # Escalador entrenado (generado)
├── model_info.pkl      # Información del modelo (generado)
├── confusion_matrix.png # Visualización del rendimiento (generada)
└── README.md           # Este archivo
```

## Instalación y Configuración

### Opción 1: Ejecución con Docker (Recomendado)

#### 1. Entrenar el Modelo
```bash
python train_model.py
```

#### 2. Construir la Imagen Docker
```bash
docker build -t breast-cancer-api .
```

#### 3. Ejecutar el Contenedor
```bash
# Ejecutar en primer plano
docker run -p 5001:5000 breast-cancer-api

# Ejecutar en segundo plano
docker run -d -p 5001:5000 --name cancer-api breast-cancer-api
```

#### 4. Gestión del Contenedor
```bash
# Ver logs
docker logs breast-cancer-api

# Detener contenedor
docker stop breast-cancer-api

# Eliminar contenedor
docker rm breast-cancer-api
```

### Opción 2: Ejecución Local

#### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

#### 2. Entrenar el Modelo
```bash
python train_model.py
```

#### 3. Ejecutar la API
```bash
python app.py
```

La API estará disponible en `http://localhost:5000`

## Endpoints de la API

### GET `/`
**Descripción**: Endpoint principal que confirma que la API está funcionando

**Respuesta de ejemplo**:
```json
{
  "status": "success",
  "message": "API de predicción de cáncer de mama lista",
  "endpoints": {
    "predict": "/predict (POST)",
    "info": "/info (GET)"
  }
}
```

### GET `/info`
**Descripción**: Devuelve información sobre el modelo cargado

**Respuesta de ejemplo**:
```json
{
  "model_type": "RandomForestClassifier",
  "preprocessing": "StandardScaler",
  "n_features": 30,
  "target_names": ["malignant", "benign"],
  "feature_names": ["mean radius", "mean texture", ...]
}
```

### POST `/predict`
**Descripción**: Realiza predicciones sobre nuevas muestras

**Entrada esperada**:
```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}
```

**Respuesta de ejemplo**:
```json
{
  "status": "success",
  "prediction": 0,
  "class_name": "malignant",
  "confidence": 0.97
}
```

## Pruebas de la API

### Pruebas Automatizadas
```bash
python test_api.py
```

### Pruebas con cURL

#### Verificar estado de la API:
```bash
curl http://localhost:5000/
```

#### Obtener información del modelo:
```bash
curl http://localhost:5000/info
```

#### Realizar una predicción:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
  }'
```

#### Probar manejo de errores:
```bash
# JSON inválido
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [1, 2, 3]}'

# Número incorrecto de características
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3, 4, 5]}'
```

## Características del Dataset

El modelo utiliza el **Dataset de Cáncer de Mama de Wisconsin** con 30 características numéricas extraídas de imágenes digitalizadas:

### Características principales (10 básicas):
1. **Radio**: Distancia desde el centro hasta los puntos del perímetro
2. **Textura**: Desviación estándar de los valores de escala de grises
3. **Perímetro**: Perímetro del núcleo celular
4. **Área**: Área del núcleo
5. **Suavidad**: Variación local en las longitudes de radio
6. **Compacidad**: (perímetro² / área) - 1.0
7. **Concavidad**: Severidad de las porciones cóncavas del contorno
8. **Puntos cóncavos**: Número de porciones cóncavas del contorno
9. **Simetría**: Simetría del núcleo
10. **Dimensión fractal**: "aproximación de línea costera" - 1

### Valores calculados:
Para cada característica se incluyen 3 valores:
- **Media** (mean)
- **Error estándar** (error)  
- **Valor extremo** (worst) - media de los tres valores más grandes

**Total**: 10 características × 3 valores = **30 características**

## Arquitectura del Modelo

### Preprocesamiento: StandardScaler
- **Propósito**: Normalizar características con diferentes escalas
- **Ejemplo**: Radio (~6-28) vs Área (~143-2501)
- **Beneficios**: 
  - Mejora convergencia del modelo
  - Evita dominancia de características con valores grandes
  - Aumenta precisión y estabilidad

### Algoritmo: RandomForestClassifier
- **Configuración**: 100 árboles, profundidad máxima 10
- **Ventajas**: Robusto contra overfitting, maneja datos no lineales
- **Salidas**: Predicción binaria (0=maligno, 1=benigno) + probabilidades

### Arquitectura de Clases (train_model.py)

#### `DataLoader`
```python
# Responsable de cargar y preparar el dataset
- load_dataset(): Carga datos de sklearn
- get_dataset_info(): Información detallada del dataset
```

#### `ModelTrainer` 
```python
# Maneja entrenamiento y evaluación
- prepare_data(): División train/test
- train_model(): Entrena RandomForest con escalado
- evaluate_model(): Métricas y visualizaciones
```

#### `ModelSaver`
```python
# Persistencia de componentes
- save_model_components(): Guarda modelo, escalador e info
```

#### `ModelPredictor` (predict_model.py)
```python
# Carga modelo y realiza predicciones
- load_model(): Carga componentes guardados
- validate_input(): Valida formato de entrada
- predict(): Realiza predicción con escalado
```

## Rendimiento del Modelo

### Métricas de Evaluación
- **Precisión**: > 97% en conjunto de prueba
- **Recall**: Alta sensibilidad para detectar casos malignos
- **Especificidad**: Alta capacidad para identificar casos benignos
- **F1-Score**: Balance óptimo entre precisión y recall

### Visualizaciones Generadas
- **Matriz de Confusión**: `confusion_matrix.png`
- **Reporte de Clasificación**: Métricas detalladas por clase

## Manejo de Errores y Validaciones

### Validaciones de Entrada
- Verificación de formato JSON válido
- Validación de clave `"features"` requerida
- Comprobación de 30 características exactas
- Validación de valores numéricos

### Códigos de Error HTTP
- **400 Bad Request**: Entrada inválida
- **404 Not Found**: Endpoint no existe
- **405 Method Not Allowed**: Método HTTP incorrecto
- **500 Internal Server Error**: Error del servidor

### Ejemplo de Respuesta de Error
```json
{
  "error": "Se esperan 30 características, se recibieron 5"
}
```

## Tecnologías Utilizadas

### Backend
- **Flask 2.3.3**: Framework web para API REST
- **scikit-learn 1.3.0**: Machine Learning y preprocesamiento
- **joblib 1.3.2**: Serialización de modelos
- **NumPy 1.24.3**: Operaciones numéricas

### Visualización y Análisis
- **matplotlib 3.7.2**: Gráficos y visualizaciones
- **seaborn 0.12.2**: Visualizaciones estadísticas avanzadas
- **pandas 2.0.3**: Manipulación de datos

### Contenerización
- **Docker**: Contenerización de la aplicación
- **Python 3.9**: Imagen base del contenedor

### Testing
- **requests 2.31.0**: Cliente HTTP para pruebas

## Seguridad y Mejores Prácticas

### Validación Robusta
- Validación estricta de tipos de datos
- Verificación de rangos de entrada
- Manejo seguro de excepciones
- Logging detallado de errores

### Arquitectura
- **Principio de Responsabilidad Única**: Cada clase tiene una función específica
- **Principio KISS**: Código simple y mantenible
- **Separación de responsabilidades**: Lógica de negocio separada de API
- **Configuración externa**: Variables de entorno para producción

## Casos de Uso

### 1. Diagnóstico Médico Asistido
- Apoyo en análisis de biopsias
- Detección temprana de tumores malignos
- Reducción de falsos negativos

### 2. Investigación Médica
- Análisis de patrones en tumores
- Validación de nuevos biomarcadores
- Estudios epidemiológicos

### 3. Educación Médica
- Entrenamiento de estudiantes de medicina
- Simulación de casos clínicos
- Validación de diagnósticos

## Extensiones Futuras

### Mejoras del Modelo
- Implementación de otros algoritmos (XGBoost, SVM, Neural Networks)
- Optimización de hiperparámetros con Grid Search
- Validación cruzada para mejor estimación de rendimiento
- Feature engineering para nuevas características

### Mejoras de la API
- Autenticación y autorización (JWT)
- Rate limiting para prevenir abuso
- Versionado de API (/v1/, /v2/)
- Documentación interactiva con Swagger/OpenAPI
- Métricas de monitoreo con Prometheus
- Logging estructurado con ELK Stack

### Infraestructura
- Orquestación con Docker Compose
- Despliegue en Kubernetes
- CI/CD con GitHub Actions
- Bases de datos para logging de predicciones
- Load balancing para alta disponibilidad

## Licencia

Este proyecto es de uso académico y educativo.

## Contacto

Para preguntas técnicas o colaboraciones, contactar al equipo de desarrollo.
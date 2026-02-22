# 🤖 Proyecto de Inteligencia Artificial y Machine Learning

## 📋 Descripción del Proyecto

Este proyecto es una colección completa de implementaciones prácticas de Inteligencia Artificial y Aprendizaje Automático (Machine Learning), desarrollado como parte de la asignatura **Tecnologías Emergentes en Desarrollo de Software** en la maestría de la Universidad CESUMA. El proyecto cubre los fundamentos, aplicaciones, algoritmos y frameworks más importantes en el campo del ML.

## 🎯 Objetivos

- Demostrar el dominio de conceptos fundamentales de Machine Learning
- Implementar modelos de ML desde cero con código bien documentado
- Comparar diferentes algoritmos y técnicas de ML
- Aplicar ML a problemas reales en diversos sectores
- Utilizar frameworks profesionales para desarrollo de modelos

## 📂 Estructura del Proyecto

```
ProyectoInteligenciaArtificial/
│
├── 4_1_fundamentos_ml.py           # Tutorial de Regresión Lineal
├── 4_2_aplicacion_salud.py         # Aplicación en Sector Salud
├── 4_3_comparacion_algoritmos.py   # Comparación de Algoritmos
├── 4_4_framework_sklearn.py        # Proyecto con Scikit-learn
├── README.md                        # Este archivo
└── WORK.md                          # Documentación detallada
```

## 📚 Contenido del Proyecto

### 🔹 4.1 Fundamentos de Inteligencia Artificial y Aprendizaje Automático
**Archivo:** `4_1_fundamentos_ml.py`

Tutorial completo de Regresión Lineal utilizando el **California Housing Dataset**.

**Características:**
- Carga y exploración de datos
- Análisis exploratorio con visualizaciones
- Preprocesamiento de datos
- Entrenamiento de modelo de regresión lineal
- Evaluación con métricas (MSE, RMSE, MAE, R²)
- Predicciones de ejemplo

**Temas cubiertos:**
- Regresión lineal simple
- División train/test
- Métricas de evaluación
- Visualización de resultados

---

### 🔹 4.2 Aplicaciones y Casos de Uso de IA en Desarrollo de Software
**Archivo:** `4_2_aplicacion_salud.py`

**Sector:** Salud  
**Problema:** Predicción de riesgo de diabetes

Sistema completo de diagnóstico médico usando **Pima Indians Diabetes Database**.

**Características:**
- Proceso completo de Data Science (end-to-end)
- Recolección y limpieza de datos médicos
- Análisis exploratorio profundo
- Modelo: Random Forest Classifier
- Validación cruzada
- Importancia de características
- Sistema de predicción para pacientes individuales

**Aplicación real:**
- Diagnóstico temprano de diabetes
- Evaluación de riesgo por paciente
- Asistencia en decisiones clínicas

---

### 🔹 4.3 Algoritmos y Técnicas de Machine Learning
**Archivo:** `4_3_comparacion_algoritmos.py`

Análisis comparativo de **3 algoritmos principales** de Machine Learning:

1. **Regresión Logística** (Clasificación)
2. **SVM - Support Vector Machine** (Clasificación)
3. **K-Means** (Clustering)

**Datasets utilizados:**
- Iris Dataset (clasificación)
- Wine Dataset (clasificación y clustering)

**Comparación incluye:**
- Accuracy, Precision, Recall, F1-Score
- Tiempo de entrenamiento y predicción
- Validación cruzada
- Visualizaciones comparativas
- Análisis de fortalezas y debilidades

---

### 🔹 4.4 Herramientas y Frameworks de IA y Machine Learning
**Archivo:** `4_4_framework_sklearn.py`

**Framework:** Scikit-learn  
**Proyecto:** Sistema de Reconocimiento de Dígitos Escritos a Mano

**Justificación de Scikit-learn:**
- Biblioteca más popular de ML en Python
- API consistente y bien documentada
- Amplia variedad de algoritmos
- Ideal para prototipado y producción

**Características del proyecto:**
- Modelo: Red Neuronal Multi-capa (MLPClassifier)
- Dataset: MNIST (dígitos 0-9)
- Optimización de hiperparámetros (GridSearchCV)
- PCA para visualización
- Serialización de modelos
- Demostración de predicciones en tiempo real

**Técnicas avanzadas:**
- Normalización de datos
- Optimización con GridSearchCV
- Early stopping
- Validación cruzada
- Análisis de errores

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
# Clonar o descargar el repositorio
cd ProyectoInteligenciaArtificial

# Instalar dependencias
pip install numpy pandas matplotlib seaborn scikit-learn joblib scipy
```

**Dependencias principales:**
- `numpy` - Operaciones numéricas
- `pandas` - Manipulación de datos
- `matplotlib` - Visualización
- `seaborn` - Visualización estadística
- `scikit-learn` - Machine Learning
- `joblib` - Serialización de modelos

### Instalación rápida (una línea)

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib scipy
```

---

## ▶️ Cómo Ejecutar

### Ejecución individual de scripts

```bash
# 4.1 - Tutorial de Regresión Lineal
python 4_1_fundamentos_ml.py

# 4.2 - Aplicación en Salud (Diabetes)
python 4_2_aplicacion_salud.py

# 4.3 - Comparación de Algoritmos
python 4_3_comparacion_algoritmos.py

# 4.4 - Framework Scikit-learn (Dígitos)
python 4_4_framework_sklearn.py
```

### Ejecución de todos los scripts

```bash
# En Linux/Mac
python 4_1_fundamentos_ml.py && python 4_2_aplicacion_salud.py && python 4_3_comparacion_algoritmos.py && python 4_4_framework_sklearn.py

# En Windows PowerShell
python 4_1_fundamentos_ml.py; python 4_2_aplicacion_salud.py; python 4_3_comparacion_algoritmos.py; python 4_4_framework_sklearn.py
```

---

## 📊 Archivos Generados

Cada script genera visualizaciones y modelos:

### 4.1 - Fundamentos ML
- `4_1_exploracion_datos.png` - Análisis exploratorio
- `4_1_evaluacion_modelo.png` - Métricas del modelo

### 4.2 - Aplicación Salud
- `4_2_exploracion_datos.png` - Distribución de características
- `4_2_correlacion.png` - Matriz de correlación
- `4_2_evaluacion_modelo.png` - ROC, matriz de confusión

### 4.3 - Comparación Algoritmos
- `4_3_datasets_visualizacion.png` - Visualización de datasets
- `4_3_comparacion_algoritmos.png` - Comparación completa

### 4.4 - Framework Scikit-learn
- `4_4_dataset_ejemplos.png` - Ejemplos del dataset
- `4_4_pca_visualizacion.png` - PCA 2D
- `4_4_entrenamiento_curvas.png` - Curvas de pérdida
- `4_4_evaluacion_modelo.png` - Métricas detalladas
- `4_4_predicciones_ejemplo.png` - Ejemplos de predicciones
- `4_4_modelo_digitos.pkl` - Modelo entrenado (serializado)
- `4_4_scaler.pkl` - Escalador (serializado)

---

## 🎓 Conceptos de Machine Learning Cubiertos

### Algoritmos
- ✅ Regresión Lineal
- ✅ Regresión Logística
- ✅ Support Vector Machines (SVM)
- ✅ Random Forest
- ✅ K-Means Clustering
- ✅ Redes Neuronales (MLP)

### Técnicas
- ✅ Train/Test Split
- ✅ Validación Cruzada
- ✅ Normalización de datos (StandardScaler)
- ✅ Reducción de dimensionalidad (PCA)
- ✅ Optimización de hiperparámetros (GridSearchCV)
- ✅ Early Stopping
- ✅ Preprocesamiento de datos

### Métricas
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ MSE, RMSE, MAE, R²
- ✅ Matriz de Confusión
- ✅ Curva ROC y AUC
- ✅ Silhouette Score
- ✅ Davies-Bouldin Index

---

## 📖 Documentación Adicional

Consulta el archivo [`WORK.md`](WORK.md) para:
- Explicación detallada de cada punto
- Metodología utilizada
- Resultados esperados
- Análisis de resultados
- Conclusiones

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión Recomendada | Uso |
|-----------|-------------------|-----|
| Python | 3.8+ | Lenguaje principal |
| NumPy | 1.21+ | Operaciones numéricas |
| Pandas | 1.3+ | Manipulación de datos |
| Scikit-learn | 1.0+ | Machine Learning |
| Matplotlib | 3.4+ | Visualización |
| Seaborn | 0.11+ | Visualización estadística |

---

## 📈 Resultados Esperados

### 4.1 - Regresión Lineal
- **R² Score:** ~0.60-0.70 (60-70% de varianza explicada)
- **RMSE:** ~$50,000-$70,000 (error promedio)

### 4.2 - Predicción de Diabetes
- **Accuracy:** ~75-80%
- **Recall:** ~70-75% (detección de casos positivos)
- **AUC-ROC:** ~0.80-0.85

### 4.3 - Comparación de Algoritmos
- **Regresión Logística:** ~95% accuracy (Iris)
- **SVM:** ~96-98% accuracy (Iris)
- **K-Means:** Silhouette Score ~0.5-0.6

### 4.4 - Reconocimiento de Dígitos
- **Accuracy:** ~95-98%
- **Velocidad:** >1000 predicciones/segundo

---

## 🔍 Características Destacadas

### Código de Calidad
- ✅ Código limpio y bien documentado
- ✅ Comentarios explicativos en español
- ✅ Estructura modular y reutilizable
- ✅ Manejo de errores
- ✅ Seguimiento de mejores prácticas

### Visualizaciones
- ✅ Gráficos profesionales con Matplotlib y Seaborn
- ✅ Matrices de correlación
- ✅ Curvas de aprendizaje
- ✅ Comparaciones visuales
- ✅ Análisis de resultados

### Documentación
- ✅ README completo
- ✅ WORK.md con análisis detallado
- ✅ Comentarios en código
- ✅ Outputs descriptivos en consola

---

## 🤝 Contribuciones

Este es un proyecto académico. Si deseas contribuir o reportar problemas:

1. Fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit de cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📝 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

## 👤 Autor

**Proyecto de Inteligencia Artificial**  
Universidad CESUMA  
Asignatura: Tecnologías Emergentes en Desarrollo de Software  
Fecha: 2026

---

## 🌟 Agradecimientos

- Dataset providers: UCI Machine Learning Repository
- Scikit-learn documentation and community
- Python scientific computing community

---

## 📞 Soporte

Para preguntas o problemas:
- Revisa la documentación en `WORK.md`
- Verifica que todas las dependencias estén instaladas
- Asegúrate de usar Python 3.8+

---

## 🚀 Próximos Pasos

- Implementar más algoritmos (XGBoost, LightGBM)
- Agregar Deep Learning con TensorFlow/PyTorch
- Crear API REST para modelos
- Implementar despliegue en producción
- Agregar más datasets y casos de uso

---

**¡Disfruta explorando el mundo del Machine Learning! 🎉**

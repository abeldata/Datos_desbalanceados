# Datos_desbalanceados

## En obras

markdown
Copy code
# Datos Desbalanceados

Este proyecto se enfoca en abordar el problema de conjuntos de datos desbalanceados mediante técnicas de remuestreo y modelado de machine learning.

## Requisitos

A continuación se presentan las librerías necesarias para ejecutar este proyecto. Asegúrate de tenerlas instaladas en tu entorno.

```bash
streamlit==X.X.X
pandas==X.X.X
numpy==X.X.X
matplotlib==X.X.X
seaborn==X.X.X
scikit-learn==X.X.X
imbalanced-learn==X.X.X
```
 
## Funcionalidades
mostrar_resultados(y_test, pred_y): Función para mostrar la matriz de confusión y calcular métricas como precisión, recall y F1-score.
base_line(X_train, X_test, y_train, y_test, auto): Modelo base utilizando regresión logística.
model_balanced(X_train, X_test, y_train, y_test): Modelo con balanceo de clases usando regresión logística.
model_subsampling(X_train, X_test, y_train, y_test): Modelado con submuestreo de la clase mayoritaria.
model_oversampling(X_train, X_test, y_train, y_test): Modelado con sobremuestreo de la clase minoritaria.
model_Smote_Tomek(X_train, X_test, y_train, y_test): Modelado con resampling utilizando SMOTE-Tomek.
model_Ensamble_Balanceo(X_train, X_test, y_train, y_test): Ensamble de modelos con balanceo de clases.

## Estructura del Repositorio

```bash
.
├── data/                 # Archivos de datos
│   ├── spam.csv
│   ├── adult.csv
│   ├── bank_marketing.csv
│   └── creditcard_fraud.csv
├── README.md             # Documentación del proyecto
├── datos_desbalanceados.py # Código principal
├── requirements.txt      # Lista de librerías requeridas
└── imagenes/             # Directorio de imágenes utilizadas
    └── comic.jpg
```
## Uso
Elige un conjunto de datos desde el panel lateral y selecciona una de las opciones para evaluar diferentes modelos y técnicas de remuestreo.


## Créditos
Este proyecto utiliza las librerías de Python, incluyendo Streamlit, Pandas, NumPy, Matplotlib, Seaborn y Scikit-learn. Agradecimientos a la comunidad de desarrolladores de estas librerías.

# Datos Desbalanceados
Este repositorio se enfoca en crear una pequeña App para abordar de manera interactiva el estudio de conjuntos de datos desbalanceados , mediante técnicas de remuestreo y  ML , sobre datasets con diferentes proporciones de desbalance.

## Demo App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abeldata-datos-desbalanceados.streamlit.app/)

## Requisitos
A continuación se presentan las librerías necesarias para ejecutar este proyecto. Asegúrate de tenerlas instaladas en tu entorno.
```bash
pandas==2.0.2
numpy==1.22.4
matplotlib==3.7.1
seaborn==0.12.2
streamlit==1.23.1
scikit-learn==1.2.2
imbalanced-learn==0.11.0
```
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
     └── desbalance.jpg
```
## Uso
Elige un conjunto de datos desde el panel lateral y selecciona una de las opciones para evaluar diferentes modelos y técnicas de remuestreo.

## Créditos
Este proyecto utiliza las librerías de Python, incluyendo Streamlit, Pandas, NumPy, Matplotlib, Seaborn y Scikit-learn.<b /> Agradecimientos a la comunidad de desarrolladores de estas librerías.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


from sklearn.metrics import confusion_matrix , precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
 
from pylab import rcParams
 
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
from collections import Counter





#definimos funciona para mostrar los resultados
def mostrar_resultados(y_test, pred_y):
    LABELS = ["True", "False"]
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(6, 2))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap='coolwarm');
    plt.title("Confusion matrix")
    plt.ylabel('Real class')
    plt.xlabel('Predicted class')
    plt.show()
    st.pyplot(plt)
    

    metricas= precision_recall_fscore_support(y_test, pred_y,labels=['0', '1'])

    # Crear un DataFrame con los valores obtenidos
    df_metricas = pd.DataFrame({
        'Precision': metricas[0],
        'Recall': metricas[1],
        'F1 Score': metricas[2],
        'Support': metricas[3]
    }, index=['Class 0', 'Class 1'])  # Asignar nombres de clase si es aplicable
        

    
    # Mostrar el DataFrame en Streamlit dentro de una caja
    
    st.write('Metricas') 
    st.write(df_metricas)







### Base_Line
#@st.cache_data(persist=True)
def base_line (X_train, X_test, y_train, y_test , auto):
    clf_base = LogisticRegression(C=1.0,penalty='l2',solver="newton-cg")
    clf_base.fit(X_train,y_train)
    
    if auto == 0 :
     return clf_base
    else :
     pred_y = clf_base.predict(X_test)
     mostrar_resultados(y_test, pred_y)

### Balanced
#@st.cache_data(persist=True)
def model_balanced(X_train, X_test, y_train, y_test):
    clf_balanced = LogisticRegression(C=1.0,penalty='l2',solver="newton-cg",class_weight="balanced")
    clf_balanced.fit(X_train, y_train)
    pred_y = clf_balanced.predict(X_test)
    mostrar_resultados(y_test, pred_y)
    

### Subsampling
#@st.cache_data(persist=True)
def model_subsampling(X_train, X_test, y_train, y_test):

    

    us = NearMiss(sampling_strategy=0.7, n_neighbors=3, version=2)
    X_train_res, y_train_res = us.fit_resample(X_train, y_train)

    after = dict(Counter(y_train))
    before = dict(Counter(y_train_res))
    combined_dict = {
        'After': after,
        'Before': before
    }
    

    df_combined = pd.DataFrame(combined_dict)
    
    
    
    subsampling = base_line(X_train_res, X_test, y_train_res, y_test , auto= False)
    pred_y = subsampling.predict(X_test)
    mostrar_resultados(y_test, pred_y)

    return (df_combined)


### Oversampling
#@st.cache_data(persist=True)
def model_oversampling(X_train, X_test, y_train, y_test):

    os =  RandomOverSampler(sampling_strategy=0.7)
    X_train_res, y_train_res = os.fit_resample(X_train, y_train)
    
    after = dict(Counter(y_train))
    before = dict(Counter(y_train_res))

    combined_dict = {
        'After': after,
        'Before': before
    }
    df_combined = pd.DataFrame(combined_dict)
    
    
    oversampling = base_line(X_train_res, X_test, y_train_res, y_test, auto= False)
    pred_y = oversampling.predict(X_test)
    mostrar_resultados(y_test, pred_y)

    return (df_combined)

### resampling con Smote-Tomek
#@st.cache_data(persist=True)
def model_Smote_Tomek(X_train, X_test, y_train, y_test):

    os_us = SMOTETomek(sampling_strategy=0.7)
    X_train_res, y_train_res = os_us.fit_resample(X_train, y_train)
    

    after = dict(Counter(y_train))
    before = dict(Counter(y_train_res))

    combined_dict = {
        'After': after,
        'Before': before
    }
    df_combined = pd.DataFrame(combined_dict)
    
    

    Smote_Tomek = base_line(X_train_res, X_test, y_train_res, y_test, auto= False)
    pred_y = Smote_Tomek.predict(X_test)
    mostrar_resultados(y_test, pred_y)


    return (df_combined)


### Ensamble de Modelos con Balanceo
#@st.cache_data(persist=True)
def model_Ensamble_Balanceo(X_train, X_test, y_train, y_test):
    bbc = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(), sampling_strategy='auto', replacement=False)
    bbc.fit(X_train, y_train)
    pred_y = bbc.predict(X_test)
    mostrar_resultados(y_test, pred_y)



       




st.sidebar.header('Datasets Desbalanceados')

distribuciones  = st.sidebar.selectbox ('Toys Datasets',('spam','adult','bank_marketing','creditcart_fraud'))

if distribuciones == 'spam':
    
    df_spam = pd.read_csv('data/spambase_ready.csv')
    
    
    st.sidebar.write(df_spam['class'].value_counts(normalize=True))

    y = df_spam.pop('class')
    X = df_spam.copy()
    #dividimos en sets de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=42)

    

elif distribuciones == 'adult':
    df_adult = pd.read_csv('data/adult_ready.csv')
    
    st.sidebar.write(df_adult['class'].value_counts(normalize=True))

    y = df_adult.pop('class')
    X = df_adult.copy()
    #dividimos en sets de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=42)


elif distribuciones == 'bank_marketing':
    df_bank_marketing = pd.read_csv('data/bank_marketing_ready.csv')
    
    st.sidebar.write(df_bank_marketing['class'].value_counts(normalize=True))

    y = df_bank_marketing.pop('class')
    X = df_bank_marketing.copy()
    #dividimos en sets de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=42)

    
elif distribuciones == 'creditcart_fraud':
    df_creditcard = pd.read_csv('data/creditcard_ready.csv')
    
    st.sidebar.write(df_creditcard['class'].value_counts(normalize=True))

    y = df_creditcard.pop('class')
    X = df_creditcard.copy()
    #dividimos en sets de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=42)



with st.sidebar:
    
    st.sidebar.header("Tratamientos:")

    # Pulsar el botón "Base line" automáticamente


    if st.sidebar.button("Base line" ,key='boton_Base_line', help='Modelo de regresión logística'):
        
        pass
        
    if st.sidebar.button("Balanced",key='boton_Balanceado', help='LR: class_weight= "balanced" '):
        
        pass
    
    if st.sidebar.button("Subsampling",key='boton_Subsampling', help='Submuestreo clase mayoritaria'):
        
        pass
    if st.sidebar.button("Oversampling",key='boton_Oversampling', help='Sobremuestreo de la clase minoritaria'):
        
        pass
    
    if st.sidebar.button("Smote-Tomek",key='boton_Smote-Tomek', help='Resampling con Smote-Tomek'):
        
        pass
    
    if st.sidebar.button("Ensamble con Balanceo",key='boton_Esmable', help='Bagging con DecisionTree'):
        
        pass




# Mencionar el botón
if st.session_state['boton_Base_line']:
    base_line (X_train, X_test, y_train, y_test , auto=True)


elif st.session_state['boton_Balanceado']:
    model_balanced(X_train, X_test, y_train, y_test)


elif st.session_state['boton_Subsampling']:
   subsampling = model_subsampling(X_train, X_test, y_train, y_test )
   
   st.write('Remuestreo usado ')
   st.write(subsampling)


elif st.session_state['boton_Oversampling']:
    
    oversampling = model_oversampling(X_train, X_test, y_train, y_test )
    
    st.write('Remuestreo usado ')
    st.write(oversampling)


elif st.session_state['boton_Smote-Tomek']:
    
    ST_sampling = model_Smote_Tomek(X_train, X_test, y_train, y_test )
   
    st.write('Remuestreo usado ')
    st.write(ST_sampling)

    
elif st.session_state['boton_Esmable']:
    model_Ensamble_Balanceo(X_train, X_test, y_train, y_test)



if 'accion_carga' not in st.session_state:

    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://raw.githubusercontent.com/abeldata/Datos_desbalanceados/Master/imagenes/comic.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
    st.session_state['accion_carga'] = True





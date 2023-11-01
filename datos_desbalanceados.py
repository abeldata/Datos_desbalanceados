import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score , precision_score ,recall_score ,f1_score
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
    LABELS = ["Normal", "Fraud"]
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    st.pyplot(plt)
    

    # Calcular las métricas
    acc = accuracy_score(y_test, pred_y)
    precision = precision_score(y_test, pred_y)
    recall = recall_score(y_test, pred_y)
    f1 = f1_score(y_test, pred_y)
    

    # Crear un DataFrame con las métricas con dos decimales
    data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'Value': [acc, precision, recall, f1]
    }

    df_metrics = pd.DataFrame(data)
    df_metrics['Value'] = df_metrics['Value'].map(lambda x: f'{x:.2f}')  # Redondear a 2 decimales

    # Mostrar el DataFrame en Streamlit dentro de una caja
    st.dataframe(df_metrics)







### Base_Line
@st.cache_data(persist=True)
def base_line (X_train, X_test, y_train, y_test , auto):
    clf_base = LogisticRegression(C=1.0,penalty='l2',solver="newton-cg")
    clf_base.fit(X_train,y_train)
    
    if auto == 0 :
     return clf_base
    else :
     pred_y = clf_base.predict(X_test)
     mostrar_resultados(y_test, pred_y)

### Balanced
@st.cache_data(persist=True)
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

    df_combined = pd.DataFrame(combined_dict)
    st.write(df_combined)
    
    
    
    
    subsampling = base_line(X_train_res, X_test, y_train_res, y_test , auto= False)
    pred_y = subsampling.predict(X_test)
    mostrar_resultados(y_test, pred_y)


### Oversampling
@st.cache_data(persist=True)
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
    st.write(df_combined)
    
    oversampling = base_line(X_train_res, X_test, y_train_res, y_test, auto= False)
    pred_y = oversampling.predict(X_test)
    mostrar_resultados(y_test, pred_y)

    

### resampling con Smote-Tomek
@st.cache_data(persist=True)
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
    st.write(df_combined)
    

    Smote_Tomek = base_line(X_train_res, X_test, y_train_res, y_test, auto= False)
    pred_y = Smote_Tomek.predict(X_test)
    mostrar_resultados(y_test, pred_y)


### Ensamble de Modelos con Balanceo
@st.cache_data(persist=True)
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

    if st.sidebar.button("Base line" ,key='boton_Base_line', help='Modelo de regresión logística'):
        
        pass
        
    if st.sidebar.button("Balanceado",key='boton_Balanceado', help='LR: class_wieght= "balanced" '):
        
        pass
    
    if st.sidebar.button("Subsampling",key='boton_Subsampling', help='Submuestreo clase mayoritaria'):
        
        pass
    if st.sidebar.button("Oversampling",key='boton_Oversampling', help='Sobremuestreo de la clase mayoritaria'):
        
        pass
    
    if st.sidebar.button("Smote-Tomek",key='boton_Smote-Tomek', help='Resampling con Smote-Tomek'):
        
        pass
    
    if st.sidebar.button("Ensamble con Balanceo",key='boton_Esmable', help='Bagging con DecisionTree'):
        
        pass


# Main

# Mencionar el botón
if st.session_state['boton_Base_line']:
    base_line (X_train, X_test, y_train, y_test , auto=True)

elif st.session_state['boton_Balanceado']:
    model_balanced(X_train, X_test, y_train, y_test)

elif st.session_state['boton_Subsampling']:
    model_subsampling(X_train, X_test, y_train, y_test )

elif st.session_state['boton_Oversampling']:
    model_oversampling(X_train, X_test, y_train, y_test )

elif st.session_state['boton_Smote-Tomek']:
    model_Smote_Tomek(X_train, X_test, y_train, y_test )

elif st.session_state['boton_Esmable']:
    model_Ensamble_Balanceo(X_train, X_test, y_train, y_test)


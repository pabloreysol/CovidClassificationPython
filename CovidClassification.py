import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score)
import numpy as np

st.write('''
# COVID-19 detection    
Detect if someone has COVID using machine learning and python ! 
''')

image = Image.open('C:\\Users\\pavit\\Desktop\\covid4.jpg')
st.image(image, caption='ML', use_column_width=True)

df = pd.read_csv('C:/Users/pavit/Desktop/Pablo/Proyectos/Covid/COVID19_MEXICO.csv', low_memory=False, encoding='latin-1')

st.subheader('Data information: ')
st.dataframe(df)
st.write(df.describe())

chart = st.bar_chart(df)

X = df.iloc[:, 0: 25].values
Y = df.iloc[:, -1].values
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# Get the user feature input from the user
def get_user_input():
    sexo = st.sidebar.slider('SEXO', 1, 2, 2)
    tipo_paciente = st.sidebar.slider('TIPO_PACIENTE', 0, 2, 2)
    intubado = st.sidebar.slider('INTUBADO', 0, 2, 2)
    neumonia = st.sidebar.slider('NEUMONIA', 0, 2, 2)
    origen = st.sidebar.slider('ORIGEN', 0, 2, 2)
    sector = st.sidebar.slider('SECTOR', 1, 13, 4)
    edad = st.sidebar.slider('EDAD', 0, 119, 40)
    nacionalidad = st.sidebar.slider('NACIONALIDAD', 0, 2, 1)
    embarazo = st.sidebar.slider('EMBARAZO', 0, 2, 2)
    lengua_indigena = st.sidebar.slider('HABLA_LENGUA_INDIG', 0, 2, 2)
    indigena = st.sidebar.slider('INDIGENA', 0, 2, 2)
    diabetes = st.sidebar.slider('DIABETES', 0, 2, 1)
    epoc = st.sidebar.slider('EPOC', 0, 2, 1)
    asma = st.sidebar.slider('ASMA', 0, 2, 1)
    inmusupr = st.sidebar.slider('INMUSUPR', 0, 2, 2)
    hipertension = st.sidebar.slider('HIPERTENSION', 0, 2, 2)
    otra_com = st.sidebar.slider('OTRA_COM', 0, 2, 2)
    cardiovascular = st.sidebar.slider('CARDIOVASCULAR', 0, 2, 2)
    obesidad = st.sidebar.slider('OBESIDAD', 0, 2, 2)
    renal_cronica = st.sidebar.slider('RENAL_CRONICA', 0, 2, 2)
    tabaquismo = st.sidebar.slider('TABAQUISMO', 0, 2, 2)
    otro_caso = st.sidebar.slider('OTRO_CASO', 0, 2, 2)
    clasificacion_final = st.sidebar.slider('CLASIFICACION_FINAL', 1, 7, 3)
    migrante = st.sidebar.slider('MIGRANTE', 0, 2, 2)
    uci = st.sidebar.slider('UCI', 0, 2, 2)
    estado = st.sidebar.slider('ENTIDAD_RES', 1, 32, 9)

    user_data = {'sexo': sexo, 'tipo_paciente': tipo_paciente, 'intubado': intubado,
                 'neumonia': neumonia, 'origen': origen, 'sector': sector,
                 'edad': edad, 'nacionalidad': nacionalidad, 'embarazo': embarazo,
                 'lengua_indigena': lengua_indigena, 'indigena': indigena,
                 'diabetes': diabetes, 'epoc': epoc, 'asma': asma, 'inmusupr': inmusupr,
                 'hipertension': hipertension, 'otra_com': otra_com, 'cardiovascular': cardiovascular,
                 'obesidad': obesidad, 'renal_cronica': renal_cronica, 'tabaquismo': tabaquismo,
                 'otro_caso': otro_caso, 'clasificacion_final': clasificacion_final, 'migrante': migrante,
                 'uci': uci, 'estado': estado}

    features = pd.DataFrame(user_data, index=[0])
    return features


user_input = get_user_input()

st.subheader('User input: ')
st.write(user_input)
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, y_train)

st.subheader('Model Test Accuracy Score: ')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')
prediction = RandomForestClassifier.predict(user_input)

st.subheader('Classification: ')
st.write(prediction)

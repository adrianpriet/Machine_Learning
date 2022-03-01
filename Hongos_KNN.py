'''
NAME
        Hongos_KNN.py

VERSION
        1.0

AUTHOR
        Victor Jesus Enriquez Castro  <victorec@lcg.unam.mx>
        Adrian Prieto Castellanos <adrianpc@lcg.unam.mx>

DESCRIPTION
        Este programa es un clasificador KNN que recibe como input
        un dataset con características de hongos venenosos y comestibles
        y retorna como output la clasificación del hongo (EDIBLE/POISONOUS)

CATEGORY
        Clasificador KNN
'''

#Importamos las librerias necesarias
import pandas as pd
import numpy as np
import seaborn as sns
from io import StringIO
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#Definimos los encoders
le = preprocessing.LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')

#Ingresamos las categorias de cada ejemplo
X = np.genfromtxt("../Data/Categorias_Exp.txt",delimiter=',',dtype=str, encoding=None)

#Transformamos los datos para poder utlizar el método fit
enc.fit(X)
X = enc.transform(X).toarray()

#Ingresamos las clases correspondientes
y =np.genfromtxt("../Data/Clases_Exp.txt",delimiter=',',dtype=str, encoding=None)

#Transformamos los datos para poder utlizar el método fit
le.fit(y)
y = le.transform(y)

#Separamos el dataset para el entrenamiento y la evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#Definimos la clasificación K Nearest neighbors
k = 5

#Definimos el clasificador
classifier = KNeighborsClassifier(n_neighbors=k)

#Entrenamos el clasificador con los datos de entrenamiento
classifier.fit(X_train, y_train)

#Hacemos la primer predicción del clasificador con los datos de evaluación
y_predict = classifier.predict(X_test)

#Hacemos una transformacion inversa para hacer legibles las clases
print(le.inverse_transform(y_predict)[:5])

#Imprimimos las estadisticas del clasificador
print("Accuracy: {}".format(accuracy_score(y_test, y_predict)))
print("Precision: {}".format(precision_score(y_test, y_predict, average="macro")))
print("Recall: {}".format(recall_score(y_test, y_predict, average="macro")))
print("F-score: {}".format(f1_score(y_test, y_predict, average="macro")))

#Imprimimos el reporte de clasificacion
target_names = ['EDIBLE', 'POISONOUS']
print(classification_report(y_test, y_predict, target_names=target_names))

#Computamos la matriz de confusion
cm = confusion_matrix(y_test, y_predict)
x_axis_labels = ["EDIBLE", "POISONOUS"]
y_axis_labels = ["EDIBLE", "POISONOUS"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("Prediccion")
plt.ylabel("Realidad")
plt.title('Confusion Matrix para el clasificador KNN')
plt.savefig("knncm.png", format='png', dpi=500, bbox_inches='tight')
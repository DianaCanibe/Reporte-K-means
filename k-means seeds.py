#K-Means
#Diana Cañibe Valle   A01749422
''' Implementación de K-means para identificación de tipos de semillas de trigo'''

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df=pd.read_csv('seedDataset.csv')
df.columns=['area','perimeter','compactness','length',
         'width','asymmetry','groove_lenght','seed_class']

X=df.drop('seed_class',axis=1)
y=df['seed_class']

#Gráfica de clásificación original
size=y*3
plt.scatter(X['length'],X['width'], s=size,c = y)
plt.title('Clasificación original')
plt.show()

#Split de datos
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)

traindata= (Xtrain.shape[0]/X.shape[0])*100
print('Porcentaje entrenamiento')
print(traindata)
testdata= (Xtest.shape[0]/X.shape[0])*100
print('Porcentaje prueba')
print(testdata)
#Gráfica de pastel de proporción de datos de entrenamiento y prueba
pie = pd.DataFrame({'datos': [traindata,testdata]},
                  index=['Train', 'Test'])
plot = pie.plot.pie(y='datos', figsize=(5, 5))
plt.show()

# Método del codo para determinar el num. de grupos (Elbow method)
kmeans_kwargs = {"random_state": 12}

sse = [] #Lista para los valores SSE por cada k de prueba
#Ciclo para determinar el mejor k entre 1 y 20
for k in range(1, 10): 
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(Xtrain)
    sse.append(kmeans.inertia_) #El atributo 'inertia' indica el valor de SSE 

#Gráfica del método     
plt.plot(range(1, 10), sse)
plt.xticks(range(1, 10))
plt.xlabel("Número de Grupos (Clusters)")
plt.ylabel("Suma Residual de Cuadrados (SSE)")
plt.show()

#Modelo
model = KMeans(n_clusters=4,random_state=12)
model.fit(Xtrain)

#Prediciones con el modelo (Pruebas)
ymodel = model.predict(Xtest)
print('Score:',accuracy_score(ytest, ymodel))

#Gráfica de clásificación según el modelo vs clasificación original
plt.scatter(X['length'],X['width'], s=size,c = y)
size=ymodel*30
plt.scatter(Xtest['length'],Xtest['width'], s=size, marker='^',c=ymodel)
plt.title('Clasificación del modelo')
plt.show()

#Grid Search para ajuste de hiperparámetros
from sklearn.model_selection import GridSearchCV
parameters={'init':('k-means++','random'),'n_init':[10,20]}
clf = GridSearchCV(model, parameters)
clf.fit(Xtrain,ytrain)
ymodel = clf.predict(Xtest)
print('Score con grid search:',accuracy_score(ytest, ymodel))
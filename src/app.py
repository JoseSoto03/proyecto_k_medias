from utils import db_connect
engine = db_connect()

import pandas as pd

url = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
data = pd.read_csv(url)

# Seleccionar las columnas necesarias
data = data[['Latitude', 'Longitude', 'MedInc']]

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=42)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6, random_state=42)
train['cluster'] = kmeans.fit_predict(train[['Latitude', 'Longitude', 'MedInc']])
test['cluster'] = kmeans.predict(test[['Latitude', 'Longitude', 'MedInc']])

import matplotlib.pyplot as plt

plt.scatter(train['Longitude'], train['Latitude'], c=train['cluster'], cmap='viridis')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Clusters de viviendas en California")
plt.show()

plt.scatter(test['Longitude'], test['Latitude'], c=test['cluster'], cmap='viridis', marker="x")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Clusters con datos de prueba")
plt.show()
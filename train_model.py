import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import data  # Importar el módulo data.py
import matplotlib.pyplot as plt

def bootstrap_regression(X, y, n_bootstraps=1000):
    coefs = []
    for _ in range(n_bootstraps):
        X_sample, y_sample = resample(X, y)
        model = LinearRegression().fit(X_sample, y_sample)
        coefs.append(model.coef_)
        
    return np.array(coefs)

# Cargar los datos sintéticos desde el módulo data.py
datos_sinteticos = data.datos_sinteticos

X = datos_sinteticos[:, :2]  # Velocidades y tiempos
y = datos_sinteticos[:, 2]   # Consumos

# Entrenar el modelo de regresión lineal con los datos sintéticos
model = LinearRegression().fit(X, y)

# Imprimir los coeficientes del modelo
print("Coeficientes del modelo:", model.coef_)
print("Intercepto del modelo:", model.intercept_)

# Realizar predicciones con el modelo entrenado
y_pred = model.predict(X)

# Graficar los valores reales vs los valores predichos
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', label='Valores reales', alpha=0.5)  # Valores reales en azul
plt.scatter(y, y, color='red', label='Valores predichos', alpha=0.5)     # Valores predichos en rojo
plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.title('Comparación entre valores reales y predichos')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Cargar los datos
df = pd.read_csv("data/viviendas.csv")

# 2. Separar variables
X = df.drop("precio", axis=1)
y = df["precio"]

# 3. Preprocesamiento
# Separar columnas numéricas y categóricas
columnas_numericas = ["superficie", "habitaciones", "antigüedad"]
columnas_categoricas = ["ubicacion"]

preprocesador = ColumnTransformer(transformers=[
    ("num", StandardScaler(), columnas_numericas),
    ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas)
])

# 4. Separar conjunto de entrenamiento y prueba
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar la variable objetivo (precio)
scaler_y = StandardScaler()
y_entreno = scaler_y.fit_transform(y_entreno.values.reshape(-1, 1))
y_prueba = scaler_y.transform(y_prueba.values.reshape(-1, 1))

# 5. Crear el pipeline con preprocesamiento
X_entreno = preprocesador.fit_transform(X_entreno)
X_prueba = preprocesador.transform(X_prueba)


# 6. Crear la red neuronal
modelo = Sequential()
modelo.add(Dense(10, activation='relu', input_shape=(X_entreno.shape[1],)))
modelo.add(Dense(1))  # Salida: un solo valor (precio)

# 7. Compilar el modelo
modelo.compile(optimizer=Adam(), loss='mean_squared_error')

# 8. Entrenar el modelo
modelo.fit(X_entreno, y_entreno, epochs=100, batch_size=10, validation_split=0.2)

# 9. Evaluar el modelo
loss = modelo.evaluate(X_prueba, y_prueba)
print(f"Pérdida (error cuadrático medio): {loss}")

# 10. Realizar predicciones
predicciones = modelo.predict(X_prueba[:5])

# Desnormalizar para obtener precios reales
predicciones_reales = scaler_y.inverse_transform(predicciones)

print("\nPredicciones de prueba (precios reales):")
for i, pred in enumerate(predicciones_reales):
    print(f"Ejemplo {i + 1}: ${pred[0]:,.2f}")

# Obtener predicciones para todo el conjunto de prueba
predicciones_completas = modelo.predict(X_prueba)
predicciones_desnormalizadas = scaler_y.inverse_transform(predicciones_completas)
y_prueba_desnormalizado = scaler_y.inverse_transform(y_prueba)

# Crear la gráfica
plt.figure(figsize=(8, 6))
plt.scatter(y_prueba_desnormalizado, predicciones_desnormalizadas, color='blue', alpha=0.6, label='Predicciones')
plt.plot([y_prueba_desnormalizado.min(), y_prueba_desnormalizado.max()],
         [y_prueba_desnormalizado.min(), y_prueba_desnormalizado.max()],
         'r--', label='Predicción perfecta')
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Predicciones vs Valores reales")
plt.legend()
plt.grid(True)

# Guardar la imagen
plt.savefig("resultados/predicciones_vs_reales.png")
plt.show()
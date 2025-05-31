import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =======================
# 1. Cargar datos
# =======================
df = pd.read_csv("data/viviendas.csv")
if len(df) < 50:
    raise ValueError("¡El conjunto de datos es demasiado pequeño! Genera más datos en generar_datos.py")

X = df.drop("precio", axis=1)
y = df["precio"]

columnas_numericas = ["superficie", "habitaciones", "antigüedad"]
columnas_categoricas = ["ubicacion"]

preprocesador_X = ColumnTransformer(transformers=[
    ("num", StandardScaler(), columnas_numericas),
    ("cat", OneHotEncoder(), columnas_categoricas)
])

scaler_y = StandardScaler()

# =======================
# 2. División y preprocesamiento
# =======================
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

X_entreno = preprocesador_X.fit_transform(X_entreno)
X_prueba = preprocesador_X.transform(X_prueba)

y_entreno = scaler_y.fit_transform(y_entreno.values.reshape(-1, 1))
y_prueba = scaler_y.transform(y_prueba.values.reshape(-1, 1))

# =======================
# 3. Modelo
# =======================
modelo = Sequential()
modelo.add(Dense(10, activation='relu', input_shape=(X_entreno.shape[1],)))
modelo.add(Dense(1))
modelo.compile(optimizer=Adam(), loss='mean_squared_error')
modelo.fit(X_entreno, y_entreno, epochs=100, batch_size=16, validation_split=0.2)

# =======================
# 4. Evaluación
# =======================
loss = modelo.evaluate(X_prueba, y_prueba)
print(f"Pérdida (error cuadrático medio): {loss:.4f}")

# =======================
# 5. Predicciones
# =======================
predicciones = modelo.predict(X_prueba[:5])
predicciones_reales = scaler_y.inverse_transform(predicciones)
print("\nPredicciones de prueba (precios reales):")
for i, pred in enumerate(predicciones_reales):
    print(f"Ejemplo {i + 1}: ${pred[0]:,.2f}")

# =======================
# 6. Gráfica
# =======================
predicciones_completas = modelo.predict(X_prueba)
predicciones_desnormalizadas = scaler_y.inverse_transform(predicciones_completas)
y_prueba_desnormalizado = scaler_y.inverse_transform(y_prueba)

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

os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/predicciones_vs_reales.png")
plt.close()

# =======================
# 7. Guardar modelo y escaladores
# =======================
os.makedirs("modelos", exist_ok=True)

# Guardar modelo entrenado
modelo.save("modelos/modelo_entrenado.keras")

# Guardar preprocesadores
joblib.dump(preprocesador_X, "modelos/preprocesador_X.pkl")
joblib.dump(scaler_y, "modelos/scaler_y.pkl")

print("\n✅ Modelo y preprocesadores guardados correctamente.")

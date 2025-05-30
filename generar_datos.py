import pandas as pd
import random

# Generar datos ficticios de viviendas
data = []
ciudades = ["Madrid", "Barcelona", "Sevilla", "Valencia", "Bilbao"]
for _ in range(200):
    superficie = random.randint(50, 250)
    habitaciones = random.randint(1, 6)
    ubicacion = random.choice(ciudades)
    antiguedad = random.randint(1, 50)
    
    # Fórmula básica para calcular precio:
    precio = superficie * 1000 + habitaciones * 5000 - antiguedad * 1000 + random.randint(-10000, 10000)
    
    data.append([superficie, habitaciones, ubicacion, antiguedad, precio])

# Crear y guardar el DataFrame
df = pd.DataFrame(data, columns=["superficie", "habitaciones", "ubicacion", "antigüedad", "precio"])
df.to_csv("data/viviendas.csv", index=False)

print("¡Datos generados correctamente!")

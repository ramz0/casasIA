import customtkinter as ctk
import pandas as pd
import joblib
import numpy as np
import os
import webbrowser
from tensorflow.keras.models import load_model
from PIL import ImageTk, Image

# ================
# Cargar modelo y preprocesadores
# ================
modelo = load_model("modelos/modelo_entrenado.keras")
preprocesador_X = joblib.load("modelos/preprocesador_X.pkl")
scaler_y = joblib.load("modelos/scaler_y.pkl")

# ================
# Crear ventana
# ================
ctk.set_appearance_mode("system")
# ctk.set_default_color_theme("blue")
ventana = ctk.CTk()
ventana.title("Predicción de Precio de Vivienda")
ventana.geometry("600x600")

# ================
# Elementos del formulario
# ================

# Título principal
etiqueta_titulo = ctk.CTkLabel(
    ventana,
    text="Predicción de Precio de Vivienda",
    font=ctk.CTkFont(size=22, weight="bold")
)
etiqueta_titulo.pack(pady=(20, 10))

# Descripción/instrucciones
etiqueta_descripcion = ctk.CTkLabel(
    ventana,
    text="Completa todos los campos del formulario y selecciona la ubicación.\nLuego haz clic en 'Predecir precio' para obtener el valor estimado.",
    font=ctk.CTkFont(size=14),
    wraplength=500,
    justify="center"
)
etiqueta_descripcion.pack(pady=(0, 20))


entrada_superficie = ctk.CTkEntry(ventana, placeholder_text="Superficie (m²)")
entrada_habitaciones = ctk.CTkEntry(ventana, placeholder_text="Número de habitaciones")
entrada_antiguedad = ctk.CTkEntry(ventana, placeholder_text="Antigüedad (años)")

combo_ubicacion = ctk.CTkComboBox(ventana, values=["CDMX", "Guadalajara", "Monterrey", "Puebla", "Mérida", "Tijuana"])

etiqueta_resultado = ctk.CTkLabel(ventana, text="", font=ctk.CTkFont(size=16, weight="bold"))

def mostrar_grafica():
    ruta = os.path.abspath("resultados/predicciones_vs_reales.png")
    webbrowser.open(ruta)

def predecir_precio():
    try:
        superficie = float(entrada_superficie.get())
        habitaciones = int(entrada_habitaciones.get())
        antiguedad = int(entrada_antiguedad.get())
        ubicacion = combo_ubicacion.get()

        # Crear DataFrame de entrada con nombres de columnas
        input_df = pd.DataFrame([{
            "superficie": superficie,
            "habitaciones": habitaciones,
            "ubicacion": ubicacion,
            "antigüedad": antiguedad
        }])

        datos_procesados = preprocesador_X.transform(input_df)
        prediccion_normalizada = modelo.predict(datos_procesados)
        prediccion_final = scaler_y.inverse_transform(prediccion_normalizada)[0][0]

        etiqueta_resultado.configure(text=f"Precio estimado: ${prediccion_final:,.2f}")

    except Exception as e:
        etiqueta_resultado.configure(text=f"Error: {str(e)}")
# ================
# Posicionar elementos
# ================
entrada_superficie.pack(pady=10)
entrada_habitaciones.pack(pady=10)
entrada_antiguedad.pack(pady=10)
combo_ubicacion.pack(pady=10)
etiqueta_resultado.pack(pady=20)

boton_predecir = ctk.CTkButton(
    ventana,
    text="Predecir precio",
    command=predecir_precio,
    fg_color="#FFD700",  
    hover_color="#FFC107" 
)
boton_predecir.pack(pady=10)

boton_ver_grafica = ctk.CTkButton(
    ventana,
    text="Ver gráfica de resultados",
    command=mostrar_grafica,
    fg_color="#9C27B0",  
    hover_color="#BA68C8"  
)
boton_ver_grafica.pack(pady=10)

# ================
# Ejecutar interfaz
# ================
ventana.mainloop()

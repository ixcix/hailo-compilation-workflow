"""
Script: 0_model_to_tflite.py
Descripción: Exporta modelos de Keras/TensorFlow a formato TensorFlow Lite (TFLite) 
             para su posterior compilación con el Hailo Dataflow Compiler (DFC).

"""

import tensorflow as tf
import os
import argparse
import numpy as np

# Configuración básica
# Ajustamos las rutas para que coincidan con tu estructura de carpetas
BASIC_CONFIG = {
    "model_name": "resnet_v1_18", # Nombre del modelo de entrada (si existe)
    "input_dir": "../model/",     # Donde buscarías un model.h5 o SavedModel
    "output_dir": "../model/"     # Donde dejaremos el .tflite para el siguiente paso
}



def get_model(input_path):
    """
    Intenta cargar un modelo existente. Si no lo encuentra, crea uno sintético
    para propósitos de demostración del flujo de trabajo básico.
    """
    if os.path.exists(input_path):
        print(f"[INFO] Cargando modelo existente desde: {input_path}")
        return tf.keras.models.load_model(input_path)
    else:
        print(f"[WARN] No se encontró modelo en {input_path}.")
        print("[INFO] Generando modelo sintético (ResNet50) para demostración...")
        
        # Creamos un modelo Keras estándar (ej. ResNet50) con pesos aleatorios
        # IMPORTANTE PARA HAILO: Definir un input shape estático.
        # Hailo prefiere [Batch, Height, Width, Channels] fijos, usualmente Batch=1 para inferencia.
        input_shape = (224, 224, 3)
        
        model = tf.keras.applications.ResNet50(
            weights=None, 
            input_shape=input_shape, 
            classes=1000
        )
        
        # Compilamos dummy para evitar warnings, aunque no vamos a entrenar
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

def export_to_tflite(model, output_path):
    print("[INFO] Iniciando conversión TF -> TFLite...")

    # 1. Configurar el Convertidor
    # ---------------------------------------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # NOTA EXPERTO HAILO: 
    # Aunque el TFLite Converter permite cuantizar (INT8), para el flujo de Hailo
    # generalmente preferimos exportar un TFLite en FLOAT32.
    # ¿Por qué? Porque el Hailo Dataflow Compiler (DFC) tiene su propio y potente
    # proceso de cuantización (Optimization Step) que usa datos de calibración reales.
    # Entregarle un modelo ya cuantizado por TF puede limitar las optimizaciones del DFC.
    
    # Esta opción limpia el grafo pero mantiene precisión float32 por defecto
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    
    # 2. Convertir
    # ---------------------------------------------------------
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"[ERROR] Falló la conversión: {e}")
        return

    # 3. Guardar
    # ---------------------------------------------------------
    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"[EXITO] Modelo TFLite guardado en: {output_path}")
    print(f"[INFO] Listo para el paso de Parseo (Hailo DFC).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportar modelo TF a TFLite para Hailo")
    
    # Definimos nombre de salida por defecto basado en la config
    default_output = os.path.join(
        BASIC_CONFIG["output_dir"], 
        f"{BASIC_CONFIG['model_name']}.tflite"
    )
    
    parser.add_argument("--input", type=str, default=os.path.join(BASIC_CONFIG["input_dir",BASIC_CONFIG["model_name"]]), help="Ruta al modelo Keras (.h5) o SavedModel")
    parser.add_argument("--output", type=str, default=default_output, help="Ruta de salida del archivo .tflite")
    
    args = parser.parse_args()

    # 1. Obtener modelo (Cargar o Crear)
    model = get_model(args.input)
    
    # 2. Exportar
    export_to_tflite(model, args.output)
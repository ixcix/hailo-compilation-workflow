"""
Script: 1_parse_model.py
Descripción: Convierte modelos entrenados (ONNX o TFLite) al formato intermedio de Hailo (HAR).
             Este es el primer paso oficial dentro del Hailo Dataflow Compiler (DFC).
"""

import os
import argparse

# --- 1. Configuración de Entorno y Logs  ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configurar variables de entorno para redirigir los logs de Hailo SDK y HailoRT
os.environ['HAILO_SDK_LOG_DIR'] = logs_dir
os.environ['HAILORT_LOGGER_PATH'] = logs_dir
# Opcional: Desactivar logs de archivo si solo quieres consola, pero mejor redirigirlos
# os.environ['HAILO_CLIENT_LOGS_ENABLED'] = 'false'
from hailo_sdk_client import ClientRunner

# Configuración básica por defecto
BASIC_CONFIG = {
    "input_model_dir": "../model/",
    "output_har_dir": "../model/",   # El HAR se suele guardar junto al modelo o en intermediate/
    "hw_arch": "hailo8",            # hailo8, hailo8l etc.
    "model_name": "dkf_lstm_minimal_states_1"         # Nombre modelo pp_bev_w_head_simp, dkf_lstm
}

def parse_model(model_path, output_har_path, hw_arch, start_nodes=None, end_nodes=None):
    # Extraer nombre del modelo del archivo si no se especifica otro
    net_name = os.path.splitext(os.path.basename(model_path))[0]
    
    print(f"[INFO] Inicializando ClientRunner para arquitectura: {hw_arch}")
    runner = ClientRunner(hw_arch=hw_arch)

    print(f"[INFO] Iniciando traducción (Parsing) del modelo: {model_path}")
    
    # Lógica unificada para ONNX y TFLite
    if model_path.endswith('.onnx'):
        # Parseo específico para ONNX
        runner.translate_onnx_model(
            model=model_path,
            net_name=net_name,
            start_node_names=start_nodes,
            end_node_names=end_nodes
        )
    elif model_path.endswith('.tflite'):
        # Parseo específico para TensorFlow Lite
        runner.translate_tf_model(
            model_path=model_path,
            net_name=net_name,
            start_node_names=start_nodes,
            end_node_names=end_nodes
        )
    else:
        raise ValueError(f"[ERROR] Formato no soportado: {model_path}. Use .onnx o .tflite")

    # Guardar el estado actual como archivo HAR (Hailo Archive)
    # Este archivo contiene el grafo parseado (HN) y los pesos (NPZ) en Float32
    print(f"[INFO] Guardando Hailo Archive (HAR) en: {output_har_path}")
    runner.save_har(output_har_path)
    
    print("[EXITO] Parseo completado. Ahora puedes proceder al paso de Optimización.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsear modelos (ONNX/TFLite) a formato Hailo HAR")
    
    # Path por defecto (prioridad TFLite, luego ONNX)
    default_model_path = os.path.join(BASIC_CONFIG["input_model_dir"], f"{BASIC_CONFIG['model_name']}.tflite")
    if not os.path.exists(default_model_path):
         default_model_path = os.path.join(BASIC_CONFIG["input_model_dir"], f"{BASIC_CONFIG['model_name']}.onnx")

    parser.add_argument("--model", type=str, default=default_model_path,help="Ruta al archivo del modelo de entrada (.onnx o .tflite)")
    parser.add_argument("--output", type=str, default=None, help="Ruta de salida para el archivo .har (por defecto usa el nombre del modelo en output_har_dir)")
    parser.add_argument("--hw-arch", type=str, default=BASIC_CONFIG["hw_arch"], choices=["hailo8", "hailo8l", "hailo15", "hailo10"],help="Arquitectura del hardware destino")
    # Argumentos avanzados para cortar el grafo (útil para modelos complejos)
    parser.add_argument("--start-nodes", nargs="+", default=None, help="Nombres de los nodos de inicio (opcional)")
    parser.add_argument("--end-nodes", nargs="+", default=None, help="Nombres de los nodos finales (opcional, para cortar post-procesados)")
    args = parser.parse_args()
    
    # Definir ruta de salida automática si no se provee
    if args.output is None:
        model_filename = os.path.basename(args.model)
        model_name_no_ext = os.path.splitext(model_filename)[0]

        # Quitar sufijo "_simplified" si existe
        if model_name_no_ext.endswith("_simplified"):
            model_name_no_ext = model_name_no_ext.replace("_simplified", "")

        args.output = os.path.join(
            BASIC_CONFIG["output_har_dir"],
            f"{model_name_no_ext}.har"
        )
    # Crear directorios necesarios
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Ejecutar
    parse_model(
        model_path=args.model,
        output_har_path=args.output,
        hw_arch=args.hw_arch,
        start_nodes=args.start_nodes,
        end_nodes=args.end_nodes
    )
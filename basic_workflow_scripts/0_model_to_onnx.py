"""
Script: 0_model_to_onnx.py
Descripción: Script unificado para exportar modelos (Keras/TF o PyTorch) al formato ONNX.
             El formato ONNX es preferible para modelos con capas recurrentes (LSTM/RNN)
             o estructuras complejas que TFLite no representa estáticamente.

Contexto:
    - Keras/TF: Usa 'tf2onnx' para convertir.
    - PyTorch: Usa 'torch.onnx.export' nativo.
    - Salida: Un archivo .onnx listo para el paso 1 (Hailo Parsing).
"""

import os
import argparse
import sys
import numpy as np

# --- 1. Configuración de Entorno y Directorios ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuración básica por defecto
BASIC_CONFIG = {
    "model_name": "dkf_lstm.h5",       # Nombre con extensión (h5, SavedModel, pth)
    "input_dir": os.path.join(project_root, "model"),
    "output_dir": os.path.join(project_root, "model"),
    "input_shape": (1, 10, 15) # dkf_lstm: (Batch, Seq_len, Features) -> (1,10,15); pp_bev_w_head_simp: (1,3,224,224,3)
}

def export_tf_to_onnx(input_path, output_path, opset=11):
    """Maneja la conversión de Keras/Tensorflow a ONNX"""
    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError:
        raise ImportError("Para convertir modelos TF, instala: pip install tensorflow tf2onnx")

    print(f"[INFO] Cargando modelo Keras desde: {input_path}")
    if os.path.isdir(input_path) or input_path.endswith(('.h5', '.keras')):
        model = tf.keras.models.load_model(input_path)
    else:
        print("[WARN] Modelo no encontrado.")


    print(f"[INFO] Convirtiendo a ONNX (opset {opset})...")
    # tf2onnx convierte el modelo keras y lo guarda
    spec = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=spec, 
        opset=opset,
        output_path=output_path
    )
    print(f"[EXITO] Modelo ONNX guardado en: {output_path}")

def export_torch_to_onnx(input_path, output_path, opset=11):
    """Maneja la conversión de PyTorch a ONNX"""
    try:
        import torch
        import torchvision
    except ImportError:
        raise ImportError("Para convertir modelos PyTorch, instala: pip install torch torchvision")

    print(f"[INFO] Cargando modelo PyTorch desde: {input_path}")
    
    # NOTA: Cargar un .pth arbitrario requiere conocer la clase del modelo.
    if os.path.exists(input_path):
        try:
            # Intentamos cargar asumiendo que es un jit script o similar, 
            # si es state_dict puro necesitaríamos la clase definida.
            model = torch.jit.load(input_path)
        except:
            print("[WARN] No se pudo cargar el .pth directamente (requiere definición de clase).")
    else:
        print("[WARN] Archivo no encontrado. Generando PyTorch dummy (ResNet18)...")
        model = torchvision.models.resnet18(pretrained=False)

    model.eval()
    
    # Crear input dummy para el trazo (Tracing)
    # Hailo prefiere dimensiones estáticas.
    dummy_input = torch.randn(*BASIC_CONFIG["input_shape"])

    print(f"[INFO] Exportando a ONNX (opset {opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # IMPORTANTE: Para LSTMs en Hailo, evita dynamic_axes si es posible.
        # Fija el tamaño de secuencia si tu aplicación lo permite.
        dynamic_axes=None 
    )
    print(f"[EXITO] Modelo ONNX guardado en: {output_path}")

def run_export(model_path, output_path, opset):
    # Detectar framework basado en extensión
    ext = os.path.splitext(model_path)[1].lower()
    
    # Asegurar directorio de salida con permisos (intentamos corregir si falla)
    out_dir = os.path.dirname(output_path)
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except PermissionError:
            print(f"[ERROR] No hay permisos para crear {out_dir}.")
            sys.exit(1)

    if ext in ['.h5', '.keras', '.pb'] or os.path.isdir(model_path):
        # Asumimos TensorFlow/Keras
        export_tf_to_onnx(model_path, output_path, opset)
    elif ext in ['.pth', '.pt']:
        # Asumimos PyTorch
        export_torch_to_onnx(model_path, output_path, opset)
    else:
        # Fallback por defecto (ej. si el archivo no existe, usará TF para crear dummy)
        print(f"[INFO] Extensión '{ext}' no reconocida o archivo no existe. Usando flujo TF por defecto.")
        export_tf_to_onnx(model_path, output_path, opset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportar modelos TF/PyTorch a ONNX para Hailo")
    


    default_model_path = os.path.join(BASIC_CONFIG["input_dir"], f"{BASIC_CONFIG['model_name']}")

    # Definición de argumentos en una sola línea (sin tabular la llamada)
    parser.add_argument("--model", type=str, default=default_model_path, help="Ruta del modelo de entrada (.h5, .pth)")
    parser.add_argument("--output", type=str, default=None, help="Ruta de salida del .onnx")
    parser.add_argument("--opset", type=int, default=11, help="Versión del Opset de ONNX (11 es recomendado para Hailo)")

    args = parser.parse_args()

    # Determinar nombre de salida si no se especifica
    if args.output is None:
        filename = os.path.basename(args.model)
        name_no_ext = os.path.splitext(filename)[0]
        # Si el nombre es genérico 'model', intentamos usar el de config
        final_name = name_no_ext if name_no_ext != 'model' else BASIC_CONFIG["model_name"]
        args.output = os.path.join(BASIC_CONFIG["output_dir"], f"{final_name}.onnx")

    run_export(args.model, args.output, args.opset)
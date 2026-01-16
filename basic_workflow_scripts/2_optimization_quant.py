"""
Script: 2_optimizacion_quant.py
Descripción: Realiza la optimización y cuantización de un modelo HAR utilizando el Hailo Dataflow Compiler (DFC).
             Requiere un dataset de calibración (.npy) generado previamente.
             NO tiene dependencias de OpenPCDet ni PyTorch.
"""

import os
import argparse
import numpy as np

# --- 1. Configuración de Entorno y Logs ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configurar variables de entorno para redirigir los logs de Hailo
os.environ['HAILO_SDK_LOG_DIR'] = logs_dir
os.environ['HAILORT_LOGGER_PATH'] = logs_dir

from hailo_sdk_client import ClientRunner

# Configuración básica por defecto
BASIC_CONFIG = {
    "input_model_dir": "../model/",
    "model_name": "pp_bev_w_head", # Nombre base
    "calib_data_dir": "../model/",   # Directorio donde está el .npy
    "hw_arch": "hailo8",
    "calib_set_size": 1024
}

def get_optimization_script(calib_size, batch_size=1):
    """
    Define los comandos ALS (Allocation Language Script) para la optimización.
    Basado en la configuración 'kitti' del script original.
    """
    commands = [
        f'model_optimization_flavor(optimization_level=0, compression_level=0, batch_size={batch_size})',
        f'model_optimization_config(calibration, calibset_size={calib_size}, batch_size={batch_size})',
        # Desactivar finetuning y bias correction según configuración KITTI original
        'post_quantization_optimization(finetune, policy=disabled)',
        'post_quantization_optimization(bias_correction, policy=disabled)',
        # Opcional: Descomentar si se necesita clipping específico como en Waymo
        # 'pre_quantization_optimization(activation_clipping, layers=[...], mode=percentile, clipping_values=[...])'
    ]
    return '\n'.join(commands)

def optimize_and_quantize(har_path, calib_path, output_qhar_path, hw_arch, nms_config_path=None):
    
    print(f"[INFO] Cargando modelo HAR desde: {har_path}")
    runner = ClientRunner(har=har_path)

    # 1. Cargar datos de calibración
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"[ERROR] No se encontró el archivo de calibración: {calib_path}. "
                                f"Este script no genera datos, debe proporcionarlos externamente.")
    
    print(f"[INFO] Cargando dataset de calibración desde: {calib_path}")
    calib_set = np.load(calib_path, mmap_mode='r')
    
    # Verificar dimensiones
    # El script original usaba transposición (0, 2, 3, 1), asumimos que el .npy ya viene en formato NHWC correcto
    print(f"[INFO] Shape del dataset de calibración: {calib_set.shape}")
    current_calib_size = calib_set.shape[0]

    # 2. Cargar Script de Optimización (ALS)
    print("[INFO] Aplicando configuración de optimización (Flavor level 0)...")
    model_script = get_optimization_script(calib_size=current_calib_size)
    
    # Inyectar NMS si se proporciona configuración (funcionalidad del script original)
    if nms_config_path and os.path.exists(nms_config_path):
        print(f"[INFO] Añadiendo capa de NMS Post-process desde: {nms_config_path}")
        model_script += f'\nnms_postprocess("{nms_config_path}", meta_arch=ssd, engine=nn_core)'
    
    runner.load_model_script(model_script)

    # 3. Ejecutar Optimización y Cuantización
    print("[INFO] Iniciando proceso de optimización y cuantización (esto puede tardar)...")
    runner.optimize(calib_set)

    # 4. Guardar resultado
    print(f"[INFO] Guardando modelo cuantizado (Q-HAR) en: {output_qhar_path}")
    runner.save_har(output_qhar_path)
    
    print("[EXITO] Proceso finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimizar y Cuantizar modelo HAR para Hailo")
    
    # Rutas por defecto construidas dinámicamente
    default_har_path = os.path.join(BASIC_CONFIG["input_model_dir"], f"{BASIC_CONFIG['model_name']}.har")
    default_qhar_path = os.path.join(BASIC_CONFIG["input_model_dir"], f"{BASIC_CONFIG['model_name']}.q.har")
    default_calib_path = os.path.join(BASIC_CONFIG["calib_data_dir"], f"calib_{BASIC_CONFIG['calib_set_size']}.npy")

    parser.add_argument("--har", type=str, default=default_har_path, help="Ruta al archivo .har de entrada")
    parser.add_argument("--output", type=str, default=default_qhar_path, help="Ruta de salida para el .q.har")
    parser.add_argument("--calib", type=str, default=default_calib_path, help="Ruta al archivo .npy de calibración")
    parser.add_argument("--hw-arch", type=str, default=BASIC_CONFIG["hw_arch"], choices=["hailo8", "hailo8l", "hailo15"], help="Arquitectura hardware")
    parser.add_argument("--nms-config", type=str, default=None, help="Ruta al json de configuración NMS (opcional)")

    args = parser.parse_args()

    # Validaciones básicas
    if not os.path.exists(args.har):
        print(f"[ERROR] El archivo HAR de entrada no existe: {args.har}")
        exit(1)

    # Crear directorios de salida si no existen
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Ejecutar
    optimize_and_quantize(
        har_path=args.har,
        calib_path=args.calib,
        output_qhar_path=args.output,
        hw_arch=args.hw_arch,
        nms_config_path=args.nms_config
    )
import os
import argparse
import numpy as np
import sys
import gc  # Garbage Collector
import shutil
from hailo_sdk_client import ClientRunner

# Desactivamos el crecimiento elástico (según Doc Oficial)
os.environ['HAILO_SET_MEMORY_GROWTH'] = 'False'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'False'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def deep_clean_system():
    """Limpia rastro de ejecuciones pasadas en RAM y Disco."""
    print("🧹 [CLEAN] Iniciando limpieza profunda de recursos...")
    
    # 1. Forzar recolección de basura de Python
    gc.collect()
    
    # 2. Limpiar carpetas temporales de Hailo/TF si existen
    tmp_path = os.environ.get('TMPDIR', '/tmp')
    hailo_tmps = [d for d in os.listdir(tmp_path) if 'hailo' in d or 'tf' in d]
    for d in hailo_tmps:
        full_path = os.path.join(tmp_path, d)
        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
        except:
            pass
    
    print("✅ [CLEAN] Sistema despejado.")

# --- Configuración de Logs ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

os.environ['HAILO_SDK_LOG_DIR'] = logs_dir
os.environ['HAILORT_LOGGER_PATH'] = logs_dir
os.environ['HAILO_CLIENT_LOGS_ENABLED'] = "False"

def optimize_and_quantize(har_path, calib_path, output_qhar_path, alls_path):
    # Ejecutamos limpieza justo antes de empezar lo gordo
    deep_clean_system()
    
    print(f"🚀 [INFO] Cargando modelo HAR: {har_path}")
    import keras.backend as K
    # ... dentro de optimize_and_quantize ...
    K.clear_session()
    gc.collect()
    runner = ClientRunner(har=har_path)

    if os.path.exists(alls_path):
        print(f"📜 [INFO] Cargando reglas (.alls): {alls_path}")
        runner.load_model_script(alls_path)
    else:
        print(f"❌ [ERROR] No se encuentra {alls_path}.")
        sys.exit(1)

    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"[ERROR] No se encontró: {calib_path}")
    
    # print(f"📊 [INFO] Cargando calibración (mmap_mode='r')...")
    # Usamos mmap y luego convertimos a float32 para asegurar compatibilidad
    # calib_set = np.load(calib_path, mmap_mode='r')
    print(f"📊 [INFO] Cargando calibración")
    calib_set = np.load(calib_path)
    
    # CORRECCIÓN DE MEMORIA: Transponer consume mucha RAM. 
    # Lo hacemos en bloques si es necesario, pero aquí lo optimizamos:
    if calib_set.ndim == 4 and calib_set.shape[1] == 48:
        print("⚠️ [AVISO] Detectado NCHW. Transponiendo a NHWC...")
        temp_array = np.array(calib_set)
        calib_set = np.transpose(temp_array, (0, 2, 3, 1)).astype(np.float32)
        
        # --- LIMPIEZA EXTRA ---
        del temp_array
        gc.collect() # Forzamos a Python a soltarlo
        # ----------------------
        print(f"✅ Nuevo Shape corregido: {calib_set.shape}")


    print("⚙️  [INFO] Iniciando optimización (Dataset completo)...")
    runner.optimize(calib_set)
        
    print(f"💾 [INFO] Guardando Q-HAR en: {output_qhar_path}")
    runner.save_har(output_qhar_path)
    print("✅ [EXITO] Modelo cuantizado listo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--har", type=str, required=True)
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--alls", type=str, default="pillarnest.alls")
    
    args = parser.parse_args()

    output_qhar_path = args.output if args.output else args.har.replace(".har", ".q.har")
    os.makedirs(os.path.dirname(output_qhar_path), exist_ok=True)

    optimize_and_quantize(
        har_path=args.har,
        calib_path=args.calib,
        output_qhar_path=output_qhar_path,
        alls_path=args.alls
    )
import os
import argparse
import numpy as np
import sys
from hailo_sdk_client import ClientRunner

# --- 1. Configuración de Entorno y Logs ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

os.environ['HAILO_SDK_LOG_DIR'] = logs_dir
os.environ['HAILORT_LOGGER_PATH'] = logs_dir
os.environ['HAILO_CLIENT_LOGS_ENABLED'] = "False"
os.environ['HAILO_SET_MEMORY_GROWTH'] = "False"



def optimize_and_quantize(har_path, calib_path, output_qhar_path, alls_path):
    
    print(f"🚀 [INFO] Cargando modelo HAR: {har_path}")
    runner = ClientRunner(har=har_path)

    # 1. Cargar Script de Modelo (.alls)
    # ESTO ES VITAL: Aquí está el arreglo del 'dead_layer_removal' y los 16-bits.
    if os.path.exists(alls_path):
        print(f"📜 [INFO] Cargando reglas (.alls): {alls_path}")
        runner.load_model_script(alls_path)
    else:
        print(f"❌ [ERROR] No se encuentra {alls_path}.")
        sys.exit(1)

    # 2. Cargar datos de calibración (MÉTODO DIRECTO)
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"[ERROR] No se encontró: {calib_path}")
    
    print(f"📊 [INFO] Cargando calibración (mmap_mode='r')...")
    # Cargamos como mmap para no explotar la RAM instantáneamente, aunque 
    # al transponer se cargará en memoria si es necesario.
    calib_set = np.load(calib_path, mmap_mode='r')
    
    print(f"   Shape original: {calib_set.shape}")

    # 3. Auto-corrección NCHW -> NHWC
    # PyTorch: (Batch, 48, 720, 720) -> Hailo: (Batch, 720, 720, 48)
    if calib_set.ndim == 4 and calib_set.shape[1] == 48 and calib_set.shape[3] != 48:
        print("⚠️ [AVISO] Detectado NCHW. Transponiendo a NHWC...")
        # Al hacer transpose sobre mmap, se crea una copia en RAM.
        # Si tienes 64 frames, son unos 6GB, debería entrar bien.
        calib_data_ram = np.array(calib_set)
        calib_set = np.transpose(calib_data_ram, (0, 2, 3, 1))
        print(f"✅ Nuevo Shape corregido: {calib_set.shape}")

    # 4. Ejecutar Optimización
    print("⚙️  [INFO] Iniciando optimización (Dataset completo)...")
    # try:
        # A PELO, como querías. El SDK se encargará de trocearlo internamente.
    runner.optimize(calib_set)
        
    # except Exception as e:
    #     print(f"\n❌ [CRASH] Falló la optimización.")
    #     print(f"   Error: {e}")
    #     # Si falla aquí, suele ser OOM de GPU real o el error de dead_layers (que ya arreglamos en el .alls)
    #     sys.exit(1)

    # 5. Guardar resultado
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
import os
import argparse
import numpy as np
import sys
import gc
import shutil
from hailo_sdk_client import ClientRunner

# --- Configuración de Entorno ---
os.environ['HAILO_SET_MEMORY_GROWTH'] = 'False'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'False'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def deep_clean_system():
    """Limpia rastro de ejecuciones pasadas en RAM y Disco."""
    print("🧹 [CLEAN] Iniciando limpieza profunda de recursos...")
    gc.collect()
    tmp_path = os.environ.get('TMPDIR', '/tmp')
    try:
        hailo_tmps = [d for d in os.listdir(tmp_path) if 'hailo' in d or 'tf' in d]
        for d in hailo_tmps:
            full_path = os.path.join(tmp_path, d)
            if os.path.isdir(full_path): shutil.rmtree(full_path)
            else: os.remove(full_path)
    except: pass
    print("✅ [CLEAN] Sistema despejado.")

def optimize_and_quantize(har_path, calib_path, output_qhar_path, alls_path):
    deep_clean_system()
    
    print(f"🚀 [INFO] Cargando modelo HAR: {har_path}")
    # Importar keras aquí dentro ayuda a que la limpieza de sesión sea efectiva
    import keras.backend as K
    K.clear_session()
    
    runner = ClientRunner(har=har_path)

    # 1. Cargar el Model Script (.alls)
    # Es vital que este archivo contenga comandos para PointPillars (ver nota abajo)
    if os.path.exists(alls_path):
        print(f"📜 [INFO] Cargando reglas (.alls): {alls_path}")
        runner.load_model_script(alls_path)
    else:
        print(f"⚠️ [AVISO] No se encontró {alls_path}, se procederá sin script de optimización.")

    # 2. Cargar Dataset de Calibración
    print(f"📊 [INFO] Cargando calibración de: {calib_path}")
    calib_set = np.load(calib_path)
    
    # ADAPTACIÓN POINTPILLARS: 
    # El canvas de PointPillars suele tener 64 canales (o los que definieras en el Scatter)
    # En lugar de buscar "48", buscamos si el formato es NCHW para pasarlo a NHWC (Hailo default)
    if calib_set.ndim == 4:
        # Si el eje 1 es pequeño (canales) y el 2-3 son grandes (400x400), es NCHW
        if calib_set.shape[1] < calib_set.shape[2]:
            print(f"⚠️ [AVISO] Detectado NCHW {calib_set.shape}. Transponiendo a NHWC...")
            calib_set = np.transpose(calib_set, (0, 2, 3, 1)).astype(np.float32)
            print(f"✅ Nuevo Shape corregido: {calib_set.shape}")
        else:
            print(f"✅ Formato NHWC detectado: {calib_set.shape}")

    # 3. Optimización / Cuantización
    print("⚙️  [INFO] Iniciando runner.optimize (esto puede tardar)...")
    try:
        runner.optimize(calib_set)
    except Exception as e:
        print(f"❌ [ERROR] Fallo en la optimización: {e}")
        sys.exit(1)
        
    # 4. Guardar resultado
    print(f"💾 [INFO] Guardando Q-HAR en: {output_qhar_path}")
    runner.save_har(output_qhar_path)
    print("✅ [EXITO] Modelo cuantizado listo para compilar a HEF.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--har", type=str, required=True, help="Path al .har parseado")
    parser.add_argument("--calib", type=str, required=True, help="Path al .npy de calibración")
    parser.add_argument("--alls", type=str, required=True, help="Path al .alls de PointPillars")
    parser.add_argument("--output", type=str, help="Path de salida")
    
    args = parser.parse_args()
    output_qhar_path = args.output if args.output else args.har.replace(".har", ".q.har")

    optimize_and_quantize(args.har, args.calib, output_qhar_path, args.alls)
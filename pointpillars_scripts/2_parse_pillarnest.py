import os
import argparse
import sys
from hailo_sdk_client import ClientRunner

# --- 1. Configuración de Entorno y Logs ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

os.environ['HAILO_SDK_LOG_DIR'] = logs_dir
os.environ['HAILORT_LOGGER_PATH'] = logs_dir

def parse_pointpillars(onnx_path, har_path=None):
    # 1. Configuración de nombres
    # Estos DEBEN coincidir exactamente con los que pusimos en el script de 'patch'
    output_names = ['cls_score', 'bbox_pred', 'dir_cls_pred']
    input_node = "input_canvas" # Nombre definido en el torch.onnx.export
    
    chosen_hw_arch = "hailo8l" 
    
    har_name = os.path.basename(onnx_path).replace("_simp_fixed.onnx", ".har")
    if not har_path:
        har_path = onnx_path.replace("_simp_fixed.onnx", ".har")

    print(f"🚀 [PARSE] Iniciando traducción de: {onnx_path}")
    print(f"🎯 [PARSE] Nodos de salida objetivo: {output_names}")

    # 2. Inicializar el Runner de Hailo
    runner = ClientRunner(hw_arch=chosen_hw_arch)

    # 3. Traducción de ONNX a HAR (Formato interno de Hailo)
    # Aquí es donde el SDK verifica que el grafo sea compatible
    try:
        runner.translate_onnx_model(
            onnx_path,
            har_name,
            start_node_names=[input_node], 
            end_node_names=output_names
        )
    except Exception as e:
        print(f"❌ [ERROR] Fallo en el parsing: {e}")
        sys.exit(1)

    # 4. Guardar el archivo HAR
    runner.save_har(har_path)
    print(f"✅ [SUCCESS] HAR generado exitosamente en: {har_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsea un modelo ONNX de PointPillars a formato HAR de Hailo.")
    parser.add_argument("input_onnx", help="Ruta al archivo ONNX (preferiblemente el _fixed.onnx)")
    parser.add_argument("--output", help="Ruta de destino para el .har", default=None)
    
    args = parser.parse_args()

    if not os.path.exists(args.input_onnx):
        print(f"❌ El archivo {args.input_onnx} no existe.")
        sys.exit(1)

    parse_pointpillars(args.input_onnx, args.output)
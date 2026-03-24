import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse

# --- 1. Configuración de Entorno y Logs ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

os.environ['HAILO_SDK_LOG_DIR'] = logs_dir
os.environ['HAILORT_LOGGER_PATH'] = logs_dir
os.environ['HAILO_CLIENT_LOGS_ENABLED'] = "True"
os.environ['HAILO_SET_MEMORY_GROWTH'] = "False"

from hailo_sdk_client import ClientRunner
import hailo_sdk_client
print(hailo_sdk_client.__version__)

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qhar", type=str, required=True)
    parser.add_argument("--output", type=str, required=False)
    # parser.add_argument("--alls", type=str, default="pillarnest.alls")
    
    args = parser.parse_args()

    output_hef_path = args.output if args.output else args.qhar.replace(".q.har", ".hef")
    os.makedirs(os.path.dirname(output_hef_path), exist_ok=True)

    # Load quantized model
    runner = ClientRunner(har=args.qhar)



    alls_line = ['performance_param(compiler_optimization_level=0)']
        #open('helper.alls','w').write(alls_line1)  #   !!!!
    # alls_line = [
    #     # # --- TASK 0 (Coches) ---
    #     # 'quantization_param([conv68, conv69, conv70, conv71, conv72, conv73, conv74], precision_mode=a16_w16)',
        
    #     # # --- TASK 1 (Camiones) ---
    #     # 'quantization_param([conv75, conv76, conv77, conv78, conv79, conv80, conv81], precision_mode=a16_w16)',
        
    #     # # --- TASK 2 (Autobuses) ---
    #     # 'quantization_param([conv82, conv83, conv84, conv85, conv86, conv87, conv88], precision_mode=a16_w16)',
        
    #     # # --- TASK 3 (Barreras) ---
    #     # 'quantization_param([conv89, conv90, conv91, conv92, conv93, conv94, conv95], precision_mode=a16_w16)',
        
    #     # # --- TASK 4 (Motos) ---
    #     # 'quantization_param([conv96, conv97, conv98, conv99, conv100, conv101, conv102], precision_mode=a16_w16)',
        
    #     # # --- TASK 5 (Peatones) - Nombres no secuenciales según tu log ---
    #     # 'quantization_param([conv103, conv104, conv59, conv61, conv63, conv65, conv67], precision_mode=a16_w16)',
    #     'resources_param(max_control_utilization=0.8, max_compute_utilization=0.8)',
    #     # --- Optimización General (Nivel 0 + FPS obligatorio en v3.30) ---
    #     'performance_param(compiler_optimization_level=0)'
    # ]

    # alls_line2 = [
    #     # # --- TASK 0 (Coches) ---
    #     # 'quantization_param([conv68, conv69, conv70, conv71, conv72, conv73, conv74], precision_mode=a16_w8)',
        
    #     # # --- TASK 1 (Camiones) ---
    #     # 'quantization_param([conv75, conv76, conv77, conv78, conv79, conv80, conv81], precision_mode=a16_w8)',
        
    #     # # --- TASK 2 (Autobuses) ---
    #     # 'quantization_param([conv82, conv83, conv84, conv85, conv86, conv87, conv88], precision_mode=a16_w8)',
        
    #     # # --- TASK 3 (Barreras) ---
    #     # 'quantization_param([conv89, conv90, conv91, conv92, conv93, conv94, conv95], precision_mode=a16_w8)',
        
    #     # # --- TASK 4 (Motos) ---
    #     # 'quantization_param([conv96, conv97, conv98, conv99, conv100, conv101, conv102], precision_mode=a16_w8)',
        
    #     # # --- TASK 5 (Peatones) - Nombres no secuenciales según tu log ---
    #     # 'quantization_param([conv103, conv104, conv59, conv61, conv63, conv65, conv67], precision_mode=a16_w16)',
    #     #'resources_param(max_control_utilization=0.9, max_compute_utilization=0.9)',
    #     # --- Optimización General (Nivel 0 + FPS obligatorio en v3.30) ---
    #     'performance_param(compiler_optimization_level=1)'
    # ]

    runner.load_model_script('\n'.join(alls_line)) 

    # Compile the model
    #runner.load_model_script('context_switch_param(mode=disabled)')
    compiled_model = runner.compile()

    # Save the compiled model
    with open(output_hef_path, 'wb') as f:
        f.write(compiled_model)


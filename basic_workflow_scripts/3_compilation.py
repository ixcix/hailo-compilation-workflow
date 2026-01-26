import os
import sys
import torch
import numpy as np
from pathlib import Path

# --- 1. Configuración de Entorno y Logs ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

os.environ['HAILO_SDK_LOG_DIR'] = logs_dir
os.environ['HAILORT_LOGGER_PATH'] = logs_dir


from hailo_sdk_client import ClientRunner
import hailo_sdk_client
print(hailo_sdk_client.__version__)

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)


################################# PATHS and config #################################
output_path = '../model'
hw_arch = 'hailo8l'
# Output file names
name = "pillarnest_tiny2"

q_har_name = f'{output_path}/{hw_arch}/{name}.q.har'
hef_name = f'{output_path}/{hw_arch}/{name}.hef'
# Load quantized model
runner = ClientRunner(har=q_har_name)

# Compile the model
#runner.load_model_script('context_switch_param(mode=disabled)')
compiled_model = runner.compile()

# Save the compiled model
with open(hef_name, 'wb') as f:
    f.write(compiled_model)


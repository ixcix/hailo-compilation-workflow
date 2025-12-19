import os
import sys
import torch
import gc

torch.cuda.empty_cache()
gc.collect()


import numpy as np
from pathlib import Path

from hailo_sdk_client import ClientRunner
import tensorflow as tf
# Evita que TensorFlow reserve toda la GPU
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass


import hailo_sdk_client
print(hailo_sdk_client.__version__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")
warnings.filterwarnings("ignore", category=FutureWarning, module="pcdet")

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

# Llamar al recolector de basura para liberar referencias
gc.collect()
torch.cuda.ipc_collect()

import openpcdet2hailo_utils as ohu;
import open3d_vis_utils as V


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu


def cfg_from_yaml_file_wrap(yaml_name, cfg):
    cwd = os.getcwd()
    os.chdir(openpcdet_clonedir+'/tools/')
    cfg_from_yaml_file(yaml_name, cfg)
    os.chdir(cwd)

# Replace with your OpenPCDet clone directory
openpcdet_clonedir = '/local/shared_with_docker/PointPillars/src/OpenPCDet'
sys.path.append(openpcdet_clonedir + '/tools/')

# Directory where logs should be saved
log_dir = '/local/shared_with_docker/PointPillars/logs'
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
log_file = os.path.join(log_dir, 'pp_compres.log')
logger = common_utils.create_logger(log_file)

################################# PATHS and config #################################
model = 'pointpillars'
dataset = 'kitti' #kitti, waymo or innovizone
# Hardware architecture
hw_arch='hailo8'

# Paths to model configuration and weights
yaml_name = f'/local/shared_with_docker/PointPillars/cfgs/{model}_{dataset}.yaml'
pth_name = f'/local/shared_with_docker/PointPillars/model/{model}_{dataset}.pth'

# Path to point cloud data custom
sample_pointclouds = f'/local/shared_with_docker/PointPillars/data/{dataset}/training/velodyne/'
imageset = 'train' # 'train' or 'test' para dejar val para evaluar
imageset_txt = f'/local/shared_with_docker/PointPillars/data/{dataset}/ImageSets/{imageset}.txt'

# File extension of point cloud files
if dataset == 'kitti':
    pc_file_extention = '.bin'
else:
    pc_file_extention = '.npy'




# Path to output dir for the onnx file
output_path = f'/local/shared_with_docker/PointPillars/output/{model}/{dataset}'



# Ruta al archivo de calibración
optimization_dataset = f'{output_path}/optimization_dataset'
os.makedirs(optimization_dataset, exist_ok=True)  # Create the directory if it doesn't exist

# Comprueba si ya hay ficheros .npy en el directorio
existing_files = sorted([f for f in os.listdir(optimization_dataset) if f.endswith('.npy')])
if existing_files:
    print(f"[INFO] Dataset de optimización ya existe en {optimization_dataset}. Total de archivos: {len(existing_files)}")
else:
    print(f"[INFO] No se encontró dataset de calibración. Generando archivos .npy en {optimization_dataset}...")
    # Leer lista de frames a usar desde el archivo ImageSets
    with open(imageset_txt, 'r') as f:
        frame_ids = [line.strip() for line in f.readlines()]
    print(f"[INFO] Cargando {len(frame_ids)} frames desde {imageset_txt}")
    
    
    # Cargar configuración de modelo y dataset
    cfg_from_yaml_file_wrap(yaml_name, cfg)

    demo_dataset = ohu.DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(sample_pointclouds),
        ext=pc_file_extention,
        logger=logger
    )
    logger.info(f'Total number of samples in dataset: {len(demo_dataset)}')

    # Crear modelo y cargar pesos
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # ---------------- Generar calibration set solo con frames del split ----------------
    with torch.no_grad():
        count = 0
        for frame_id in frame_ids:
            pc_path = os.path.join(sample_pointclouds, f"{frame_id}{pc_file_extention}")
            if not os.path.exists(pc_path):
                continue  # evita errores si el frame no existe

            # Cargar la nube directamente desde disco (sin depender de método interno)
            _data_dict = demo_dataset.collate_batch([demo_dataset.__getitem__(int(frame_id))])
            load_data_to_gpu(_data_dict)
            _ = model.forward(_data_dict)

            # Guardar spatial_features en formato (H, W, C)
            sample_data = np.transpose(_data_dict['spatial_features'].cpu().numpy(), (0, 2, 3, 1))[0]
            sample_path = os.path.join(optimization_dataset, f"sample_{count:04d}.npy")
            np.save(sample_path, sample_data)
            del _data_dict, sample_data
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

            count += 1
            if count >= 1024:
                break

    print(f"[INFO] Dataset de optimización con 1024 muestras generado en {optimization_dataset} con {count} muestras del split '{imageset}'.")



########### OPTIMIZATION ##############



all_lines_kitti= [   
    'post_quantization_optimization(mix_precision_search, policy=enabled, batch_size=1)'
]


all_lines_waymo = [
    
]

all_lines_innovizone = [
]


# Calibration set and optimization configuration 
if dataset == 'waymo':
      
    all_lines = all_lines_waymo
    custom_optimization = True
    nms_hailo = False

elif dataset == 'kitti':

    all_lines = all_lines_kitti
    custom_optimization = True 
    nms_hailo = False


elif dataset == 'innovizone':
 
    all_lines = all_lines_innovizone
    custom_optimization = True 
    nms_hailo = False


name = "opt_all_lines_kitti_final"

# Output file names
q_har_name = f'{output_path}/pp_bev_w_head_{name}.har'
comp_har_name = f'{output_path}/pp_bev_w_head_{name}.q.har'
#q_har_path = '/local/shared_with_docker/PointPillars/output/pointpillars/kitti/pp_bev_w_head_all_lines_kitti_2_clipp_manual3_finetune3.q.har'

try:
    del model, demo_dataset
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
except:
    pass


runner = ClientRunner(har=q_har_name)


if custom_optimization:
    runner.load_model_script('\n'.join(all_lines))


# Optimize the model and quant
runner.optimize(optimization_dataset)
# runner.analyze_noise(calib_set)
runner.save_har(comp_har_name)



torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()
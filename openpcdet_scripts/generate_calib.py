import os
import sys
import torch
import gc
import numpy as np
from pathlib import Path

import tensorflow as tf
import hailo_sdk_client
print(hailo_sdk_client.__version__)

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)


import openpcdet2hailo_utils as ohu;
import open3d_vis_utils as V


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu


# Replace with your OpenPCDet clone directory
openpcdet_clonedir = '/local/shared_with_docker/PointPillars/src/OpenPCDet'
sys.path.append(openpcdet_clonedir + '/tools/')


################################# PATHS and config #################################
model = 'pointpillars'
dataset = 'kitti' #kitti, waymo or innovizone

# Paths to model configuration and weights
yaml_name = f'/local/shared_with_docker/PointPillars/cfgs/{model}_{dataset}.yaml'
pth_name = f'/local/shared_with_docker/PointPillars/model/{model}_{dataset}.pth'

# Path to point cloud data custom
# sample_pointclouds = f'/local/shared_with_docker/PointPillars/data/{dataset}'
# demo_pointcloud = f'/local/shared_with_docker/PointPillars/data/{dataset}/'
sample_pointclouds = f'/local/shared_with_docker/PointPillars/data/{dataset}/calib_dataset'
demo_pointcloud = f'/local/shared_with_docker/PointPillars/data/{dataset}/calib_dataset/'
# sample_pointclouds = f'/local/shared_with_docker/PointPillars/data/mis_nubes'
# demo_pointcloud = f'/local/shared_with_docker/PointPillars/data/mis_nubes/'

# File extension of point cloud files
if dataset == 'kitti':
    pc_file_extention = '.bin'
else:
    pc_file_extention = '.npy'

# Path to output dir for the onnx file
output_path = f'/local/shared_with_docker/PointPillars/output/{model}/{dataset}'

# Output file names
har_name = f'{output_path}/pp_bev_w_head.har'
opt_har_name = f'{output_path}/pp_bev_w_head_opt.har'
q_har_name = f'{output_path}/pp_bev_w_head.q.har'


# Directory where logs should be saved
log_dir = '/local/shared_with_docker/PointPillarsHailoInnoviz/logs'
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
log_file = os.path.join(log_dir, 'pp_comp.log')
logger = common_utils.create_logger(log_file)


# Calibration set
if dataset == 'waymo':
    cs_size = 64
    calib_shape = (cs_size, 468, 468, 64)  
elif dataset == 'kitti':
    cs_size = 64
    calib_shape = (cs_size, 496, 432, 64)  
elif dataset == 'innovizone':
    cs_size = 8
    calib_shape = (cs_size, 468, 468, 64)  


def cfg_from_yaml_file_wrap(yaml_name, cfg):
    cwd = os.getcwd()
    os.chdir(openpcdet_clonedir+'/tools/')
    cfg_from_yaml_file(yaml_name, cfg)
    os.chdir(cwd)

def get_model(cfg, pth_name, demo_dataset):    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
    model.eval()
    return model



cfg_from_yaml_file_wrap(yaml_name, cfg)



demo_dataset = ohu.DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    root_path=Path(sample_pointclouds), ext=pc_file_extention, logger=logger
)
logger.info(f'Total number of samples: \t{len(demo_dataset)}')


model = get_model(cfg, pth_name, demo_dataset)
model.cuda()
model.eval()




# perc4stats = [50, 90, 98.6, 99.7, 99.9]

# with torch.no_grad():
#     for idx, _data_dict in enumerate(demo_dataset):        
#         # print(f"cloud #{idx}")
#         if idx >= cs_size:
#             break
        
#         _data_dict = demo_dataset.collate_batch([_data_dict])
#         load_data_to_gpu(_data_dict)
#         pred_dicts = model.forward(_data_dict)  
#         calib_set[idx] = np.transpose(_data_dict['spatial_features'].cpu().numpy(), (0,2,3,1))
#         calib_set[idx] = _data_dict['spatial_features'].cpu().numpy()
#         # print(calib_set[idx].dtype)

#         # Basic stats just to verify there's some data diversity (just in top percentile apparently...)
#         # print(f'basic stats - percentile {perc4stats} of data (@ 2d-net input)', \
#         #       np.percentile((np.abs(calib_set[idx])), perc4stats))
               
        
  

filename = '/local/shared_with_docker/calib_set_memmap.dat'

calib_set = np.memmap(filename, mode='w+', shape=calib_shape)

for idx, _data_dict in enumerate(demo_dataset):        
    # print(f"cloud #{idx}")
    if idx >= calib_shape[0]:
        break
        
    _data_dict = demo_dataset.collate_batch([_data_dict])
    load_data_to_gpu(_data_dict)
    pred_dicts = model.forward(_data_dict)
    spatial_feat = _data_dict['spatial_features'].cpu().detach().numpy()
    spatial_feat = np.transpose(spatial_feat, (0, 2, 3, 1))  # BCHW → BHWC
    calib_set[idx] = spatial_feat
    

# Reabrir como modo lectura para la optimización
calib_set = np.memmap(filename, mode='r', shape=calib_shape)


np.save(f'{output_path}/calib_{cs_size}.npy', calib_set)


# LIBERAR GPU COMPLETAMENTE DESPUÉS DE USAR TORCH 

import gc
import torch

# Eliminar explícitamente objetos que ocupan memoria
del model
del _data_dict
del spatial_feat
torch.cuda.empty_cache()

# Llamar al recolector de basura para liberar referencias
gc.collect()
torch.cuda.ipc_collect()

# Verificación opcional: imprimir memoria libre
print(f"[INFO] Memoria GPU liberada. Uso actual: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

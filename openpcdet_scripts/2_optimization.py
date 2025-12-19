import os
import sys
import torch
import gc
import numpy as np
from pathlib import Path

from hailo_platform import HEF
from hailo_sdk_client import ClientRunner
from hailo_sdk_client import InferenceContext #SdkPartialNumeric, SdkNative # 
import tensorflow as tf
tf.config.list_physical_devices('GPU')
import hailo_sdk_client
print(hailo_sdk_client.__version__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")
warnings.filterwarnings("ignore", category=FutureWarning, module="pcdet")

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)


# # Llamar al recolector de basura para liberar referencias
# gc.collect()
# torch.cuda.ipc_collect()

import openpcdet2hailo_utils as ohu;
import open3d_vis_utils as V


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu

from hailo_sdk_client import ClientRunner

# def get_snr_results(runner, out_layer_name):
#     params_statistics = runner.get_params_statistics()
#     layers = []
#     snr = []
#     for layer in runner.get_hn_model():
#         layer_snr = params_statistics.get(
#             f"{layer.name}/layer_noise_analysis/noise_results/{out_layer_name}"
#         )
#         if layer_snr is not None:
#             layers.append(layer.name_without_scope)
#             snr.append(layer_snr[0].tolist())
#     return layers, snr

def cfg_from_yaml_file_wrap(yaml_name, cfg):
    cwd = os.getcwd()
    os.chdir(openpcdet_clonedir+'/tools/')
    cfg_from_yaml_file(yaml_name, cfg)
    os.chdir(cwd)




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
sample_pointclouds = f'/local/shared_with_docker/PointPillars/data/{dataset}/training/velodyne/'
demo_pointcloud = f'/local/shared_with_docker/PointPillars/data/{dataset}/training/velodyne/'
# sample_pointclouds = f'/local/shared_with_docker/PointPillars/data/mis_nubes'
# demo_pointcloud = f'/local/shared_with_docker/PointPillars/data/mis_nubes/'

# File extension of point cloud files
if dataset == 'kitti':
    pc_file_extention = '.bin'
else:
    pc_file_extention = '.npy'

# Path to output dir for the onnx file
output_path = f'/local/shared_with_docker/PointPillars/output/{model}/{dataset}'

cs_size = 1024  # Calibration set size

# Output file names
har_name = f'{output_path}/pp_bev_w_head.har'
opt_har_name = f'{output_path}/pp_bev_w_head_opt_{cs_size}.har'
q_har_name = f'{output_path}/pp_bev_w_head_{cs_size}.q.har'
nms_config_file = '/local/shared_with_docker/PointPillars/src/pointpillars_nms_postprocess_config.json'
# Hardware architecture
hw_arch='hailo8'

# Directory where logs should be saved
log_dir = '/local/shared_with_docker/PointPillarsHailoInnoviz/logs'
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
log_file = os.path.join(log_dir, 'pp_comp.log')
logger = common_utils.create_logger(log_file)

# Ruta al archivo de calibración

calib_file_path = f'{output_path}/calib_{cs_size}.npy'

if os.path.exists(calib_file_path):
    print(f"[INFO] calib_{cs_size}.npy ya existe en {calib_file_path}. Cargando...")
    calib_set = np.load(calib_file_path, mmap_mode='r')
else:
    print(f"[INFO] calib_{cs_size}.npy no encontrado. Generando nuevo dataset de calibración...")

    cfg_from_yaml_file_wrap(yaml_name, cfg)

    demo_dataset = ohu.DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(sample_pointclouds), ext=pc_file_extention, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

      
    calib_set = np.zeros((cs_size, 496, 432, 64), dtype=np.float32)
    np.set_printoptions(precision=2)

    perc4stats = [50, 90, 98.6, 99.7, 99.9]
    with torch.no_grad():
        for idx, _data_dict in enumerate(demo_dataset):
            if idx >= cs_size:
                break
            #print(f"[INFO] Procesando nube #{idx}")
            _data_dict = demo_dataset.collate_batch([_data_dict])
            load_data_to_gpu(_data_dict)
            pred_dicts = model.forward(_data_dict)
            calib_set[idx] = np.transpose(_data_dict['spatial_features'].cpu().numpy(), (0, 2, 3, 1))

            #print(f"[DEBUG] Percentiles {perc4stats} de entrada 2D:",np.percentile(np.abs(calib_set[idx]), perc4stats))

    np.save(calib_file_path, calib_set)
    print(f"[INFO] Calib dataset guardado en: {calib_file_path}")


########### OPTIMIZATION ##############

all_lines_kitti= [
    'model_optimization_flavor(optimization_level=0, compression_level=0, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv3,conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv18], mode=percentile, clipping_values=[0.01, 99.99])',    
    # 'pre_quantization_optimization(activation_clipping, layers=[conv19], mode=percentile, clipping_values=[0.01, 99.999])',
    #'pre_quantization_optimization(activation_clipping, layers=[conv20], mode=percentile, clipping_values=[0.001, 99.999])',
    #'pre_quantization_optimization(weight_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])',
    #'quantization_param({conv*}, null_channels_cutoff_factor=1e2)',
    #clipping_values=[0.001, 99.99])', hasta ahora el mejor
   
    'post_quantization_optimization(finetune, policy=disabled)',
    'post_quantization_optimization(bias_correction,  policy=disabled)',
    
    #'resources_param(max_utilization=0.8)'

]


all_lines_waymo = [
    'model_optimization_flavor(optimization_level=2, compression_level=0, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv3,conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18], mode=percentile, clipping_values=[0.01, 99.99])',    
    'pre_quantization_optimization(activation_clipping, layers=[conv19], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv20], mode=percentile, clipping_values=[0.001, 99.999])',
    #'pre_quantization_optimization(weight_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])',
    #'quantization_param({conv*}, null_channels_cutoff_factor=1e2)',
    #clipping_values=[0.001, 99.99])', hasta ahora el mejor
   
    'post_quantization_optimization(finetune, policy=disabled)',
    'post_quantization_optimization(bias_correction,  policy=enabled)'
    
    #'resources_param(max_utilization=0.8)'
    
]

all_lines_innovizone = [
]


# Calibration set and optimization configuration 
if dataset == 'waymo':
    
    calib_shape = (cs_size, 468, 468, 64)  # Para Waymo
    all_lines = all_lines_waymo
    custom_optimization = True
    nms_hailo = False

elif dataset == 'kitti':

    calib_shape = (cs_size, 496, 432, 64)  # Para Waymo
    all_lines = all_lines_kitti
    custom_optimization = False 
    nms_hailo = False


elif dataset == 'innovizone':

    calib_shape = (cs_size, 468, 468, 64)  # Para Waymo
    all_lines = all_lines_innovizone
    custom_optimization = False 
    nms_hailo = False



if nms_hailo:
    all_lines.append(f'nms_postprocess("{nms_config_file}", meta_arch=ssd, engine=nn_core)')

# cfg_from_yaml_file_wrap(yaml_name, cfg)


runner = ClientRunner(har=har_name )

calib_set = np.load(calib_file_path, mmap_mode='r')

#print(runner.model_summary())


if custom_optimization:
    runner.load_model_script('\n'.join(all_lines))

# runner.optimize_full_precision(calib_set)

# runner.save_har(opt_har_name)


# Optimize the model and quant
runner.optimize(calib_set)
# runner.analyze_noise(calib_set)
runner.save_har(q_har_name)




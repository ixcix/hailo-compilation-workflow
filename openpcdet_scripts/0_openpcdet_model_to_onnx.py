import os
import sys
import warnings
# Filtrar inmediatamente
warnings.filterwarnings("ignore") # A veces es mejor ignorar todo si es muy ruidoso
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import numpy as np
from pathlib import Path

import onnx
from onnxsim import simplify


# Import OpenPCDet modules
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu

import openpcdet2hailo_utils as ohu
import dataset_utils as du

warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")
warnings.filterwarnings("ignore", category=FutureWarning, module="pcdet")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# --- Import Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)



################################# Setup Logger #################################

log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, '0_openpcdet_model.log')
logger = common_utils.create_logger(log_file)

################################# PATHS and Config #################################

# Configuration: Choose model and dataset
model = 'centerpoint-pillar' # Options: pointpillars, centerpoint-pillar
dataset = 'kitti'      # Options: kitti
logger.info(f'MODEL: {model}, DATASET: {dataset}')

# Paths to model configuration and weights
cfg_file = os.path.join(project_root, 'cfgs', f'{model}_{dataset}.yaml')
ckpt_file = os.path.join(project_root, 'checkpoints', f'{model}_{dataset}.pth')

# Path to output directory for the saved model structure
output_dir = os.path.join(project_root, 'output', model, dataset)
os.makedirs(output_dir, exist_ok=True)

# ONNX model filenames
onnx_name = os.path.join(output_dir, f'{model}_{dataset}.onnx')
onnx_name_simp = os.path.join(output_dir, f'{model}_{dataset}_simplified.onnx')

# Path to point cloud data
if dataset == 'kitti':
    pointclouds_dir = os.path.join(project_root, 'data', dataset, 'training', 'velodyne')
    pc_file_extension = '.bin'
else:
    pointclouds_dir = os.path.join(project_root, 'data', dataset)
    pc_file_extension = '.npy'

################################# Load Configuration and Build Model #################################

logger.info(f'-------- Loading Configuration --------')
logger.info(f'Config file: {cfg_file}')

if not os.path.exists(cfg_file):
    logger.error(f"Configuration file not found: {cfg_file}")
    sys.exit(1)

cfg_from_yaml_file(cfg_file, cfg)

logger.info(f'-------- Loading Dataset (Demo) --------')
# Create a demo dataset to verify data loading pipeline
demo_dataset = du.DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    training=False,
    root_path=Path(pointclouds_dir),
    ext=pc_file_extension,
    logger=logger
)
logger.info(f'Total number of samples: \t{len(demo_dataset)}')

logger.info(f'-------- Building Model --------')
# Build the model architecture
model = build_network(
    model_cfg=cfg.MODEL,
    num_class=len(cfg.CLASS_NAMES),
    dataset=demo_dataset
)

# Load Pre-trained weights
if os.path.exists(ckpt_file):
    logger.info(f'Loading weights from: {ckpt_file}')
    model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=True)
else:
    logger.warning(f'Checkpoint not found at {ckpt_file}. Using random weights (OK for architecture testing).')

# Set model to evaluation mode and move to GPU if available
model.eval()
if torch.cuda.is_available():
    model.cuda()
    logger.info("Model moved to CUDA.")
else:
    logger.info("CUDA not available. Model executing on CPU.")

# Print model structure (optional, can be verbose)
# print(model)

################################# Run a Sanity Test #################################

logger.info(f'-------- Running Sanity Check --------')

with torch.no_grad():
    for idx, data_dict in enumerate(demo_dataset): 
        if idx >= 2: # Test only on the first 2 samples
            break      
        
        # Prepare the batch
        data_dict = demo_dataset.collate_batch([data_dict])
        
        # CRITICAL FIX: Move data tensors to the same device as the model (GPU/CPU)
        load_data_to_gpu(data_dict)
        
        # Run inference
        pred_dicts, _ = model(data_dict)
        #print(pred_dicts)
        # Log results
        num_detections = len(pred_dicts[0]['pred_scores'])
        logger.info(f"Sample {idx}: Detected {num_detections} objects.")
        
        if num_detections > 0:
            # Show top scores
            top_scores = pred_dicts[0]['pred_scores'][:5]
            logger.info(f"Top 5 scores: {top_scores}")

        
    logger.info("Sanity check passed successfully.")


################################# Integrate with Hailo Hardware #################################

logger.info('------ Integrate with Hailo Hardware -------')
# We will offload the 2D backbone and detection head computations to the Hailo device.

### Export the 2D Backbone and Detection Head to ONNX
# Extract the 2D convolutional parts and export them to ONNX format.

# model.cpu()

#Define the module that includes backbone_2d and dense_head
pre_bev_w_head = ohu.Pre_Bev_w_Head(model)
bev_w_head = ohu.Bev_w_Head(model.backbone_2d, model.dense_head)


with torch.no_grad():
    for idx, data_dict in enumerate(demo_dataset): 
        if idx >= 1: # Test only on the first 2 samples
            break      
        
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        batch_dict_pre = pre_bev_w_head(data_dict)  # Get a sample batch_dict
        bev_out = bev_w_head(batch_dict_pre['spatial_features'])  # Get a sample batch_dict


#print(bev_out.keys())
#print(bev_out)

end_node_names = ['model/concat1', 'model/conv19', 'model/conv18', 'model/conv20']

torch.onnx.export(
    bev_w_head,
    args=(batch_dict_pre['spatial_features'],),
    f=onnx_name,
    verbose=False,
    opset_version=13,
    input_names=['spatial_features'],
    output_names= end_node_names
)

#exit()
# Simplify the ONNX model:

model = onnx.load(onnx_name)

model_simplified, check = simplify(model)

if check:
    onnx.save(model_simplified, f"{onnx_name_simp}")
    print(f"Modelo simplificado guardado en {onnx_name_simp}")
else:
    print("La simplificación del modelo falló.")



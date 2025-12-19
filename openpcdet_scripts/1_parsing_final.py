### Import Libraries and Set Paths ###

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Replace with your OpenPCDet clone directory
openpcdet_clonedir = '/local/shared_with_docker/PointPillars/src/OpenPCDet'
# sys.path.append(openpcdet_clonedir)
sys.path.append(openpcdet_clonedir + '/tools/')

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network


import onnx
from onnxsim import simplify


from hailo_sdk_client import ClientRunner
from hailo_sdk_client import InferenceContext 
import hailo_sdk_client
print(hailo_sdk_client.__version__)



# Import custom utilities for OpenPCDet and Hailo integration
import openpcdet2hailo_utils as ohu;
import open3d_vis_utils as V

################################# PATHS and config #################################
model = 'centerpoint-pillar'
dataset = 'kitti' #kitti, waymo or innovizone
print(f'MODEL: {model}, DATASET: {dataset}')

# Paths to model configuration and weights
yaml_name = f'/local/shared_with_docker/PointPillars/cfgs/{model}_{dataset}.yaml'
pth_name = f'/local/shared_with_docker/PointPillars/model/{model}_{dataset}.pth'

# Path to point cloud data custom
sample_pointclouds = f'/local/shared_with_docker/PointPillars/data/{dataset}/training/velodyne'
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

# Hardware architecture
hw_arch='hailo8'

onnx_name = f'{output_path}/pp_bev_w_head.onnx'
onnx_name_simp = f'{output_path}/pp_bev_w_head_simp.onnx'
har_name = f'{output_path}/pp_bev_w_head.har'

# Directory where logs should be saved
log_dir = '/local/shared_with_docker/PointPillars/logs'
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
log_file = os.path.join(log_dir, 'pp_to_har.log')
logger = common_utils.create_logger(log_file)


################################# Load Configuration and Build the Model #################################
cfg_from_yaml_file(yaml_name, cfg)

demo_dataset = ohu.DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    training=False,
    root_path=Path(sample_pointclouds),
    ext=pc_file_extention,
    logger=logger
)

logger.info(f'Total number of samples: \t{len(demo_dataset)}')

# Build the model and load pretrained weights
model = build_network(
    model_cfg=cfg.MODEL,
    num_class=len(cfg.CLASS_NAMES),
    dataset=demo_dataset
)
model.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
model.eval()
print(model)


################################# Run a Sanity Test #################################

def get_model(cfg, pth_name, demo_dataset):    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
    model.eval()
    return model

def cfg_from_yaml_file_wrap(yaml_name, cfg):
    cwd = os.getcwd()
    os.chdir(openpcdet_clonedir+'/tools/')
    cfg_from_yaml_file(yaml_name, cfg)
    os.chdir(cwd)


cfg_from_yaml_file_wrap(yaml_name, cfg)

logger.info('----------------------------- OPENPCDET -----------------------------')
demo_dataset = ohu.DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    root_path=Path(demo_pointcloud), ext=pc_file_extention, logger=logger
)
logger.info(f'Total number of samples: \t{len(demo_dataset)}')

model = get_model(cfg, pth_name, demo_dataset)
# print(model)



logger.info('------ Quick Demo of OpenPCDet''s PointPillarsCPU -------')
model_cpu = ohu.PointPillarsCPU(model)

with torch.no_grad():
    for idx, data_dict in enumerate(demo_dataset): 
        if idx >= 2:
            break      
        data_dict = demo_dataset.collate_batch([data_dict])
        
        
            
        pred_dicts1 = model_cpu(data_dict)
    
        
            
        print(f"Numero de pred_scores: {len(pred_dicts1[0][0]['pred_scores'])}")
        print(f"Pred_scores (7): {pred_dicts1[0][0]['pred_scores'][:20]}")

        print(pred_dicts1[0][0].keys())

        
        # ----- Aquí añadimos la parte de visualización -----
        # 1) Averigua el path de la nube
        sample_file = demo_dataset.sample_file_list[idx]  # p.ej. "000123.bin" o ".npy"
        cloud_path = os.path.join(demo_pointcloud, sample_file)

        # 2) Cárgala en NumPy
        points = V.load_point_cloud(cloud_path)

        # 3) Extrae las predicciones
        pred = pred_dicts1[0][0]
        boxes  = pred['pred_boxes'].cpu().detach().numpy()
        scores = pred['pred_scores'].cpu().detach().numpy()
        labels = pred['pred_labels'].cpu().detach().numpy()

        print(f"Frame {idx} → {len(boxes)} cajas")
        demo = False
        if demo:
            # 4) Dibuja
            V.draw_scenes(
                points=points,
                ref_boxes=boxes,
                ref_scores=scores,
                ref_labels=labels
            )
    

        break


logger.info('------ Integrate with Hailo Hardware -------')
################################# Integrate with Hailo Hardware#################################
# We will offload the 2D backbone and detection head computations to the Hailo device.

### Export the 2D Backbone and Detection Head to ONNX
# Extract the 2D convolutional parts and export them to ONNX format.

model.cpu()

#Define the module that includes backbone_2d and dense_head
bev_w_head = ohu.Bev_w_Head(model.backbone_2d, model.dense_head)

end_node_names = ['model/concat1', 'model/conv19', 'model/conv18', 'model/conv20']

torch.onnx.export(
    bev_w_head,
    args=(data_dict['spatial_features'],),
    f=onnx_name,
    verbose=False,
    opset_version=13,
    input_names=['spatial_features'],
    output_names= end_node_names
)


# Simplify the ONNX model:

# Cargar el modelo ONNX
model = onnx.load(onnx_name)

# Simplificar el modelo
model_simplified, check = simplify(model)

if check:
    # Guardar el modelo simplificado
    onnx.save(model_simplified, f"{onnx_name_simp}")
    print(f"Modelo simplificado guardado en {onnx_name_simp}")
else:
    print("La simplificación del modelo falló.")


runner = ClientRunner(hw_arch=hw_arch)

hn, npz = runner.translate_onnx_model(onnx_name_simp, end_node_names = end_node_names)

#print(f'hn: {hn}')


# Save the translated model
runner.save_har(har_name)
# runner = ClientRunner(har=har_name)


# # Aquí forzamos la carga manual del modelo
runner._sdk_backend.update_fp_model(runner._sdk_backend.model)
print("fp_model is None:", runner._sdk_backend.fp_model is None)


class Bev_W_Head_Hailo(torch.nn.Module):
    """ Drop-in replacement to the sequence of original "backbone-2d" and "dense_head" modules, accepting and returning dictionary,
        while under the hood using Hailo [emulator] implementation for the 2D CNN part, accepting and returning tensors I/O
    """
    
    def __init__(self, runner, emulate_quantized=False, use_hw=False, emulate_optimized=False, generate_predicted_boxes=None, emulate_native=False):
        super().__init__()
        self._runner = runner
        self.generate_predicted_boxes = generate_predicted_boxes
        
        if use_hw:
            context_type = InferenceContext.SDK_HAILO_HW
        elif emulate_quantized:
            context_type = InferenceContext.SDK_QUANTIZED 
        elif emulate_native:
            context_type = InferenceContext.SDK_NATIVE
        else:
            context_type = InferenceContext.SDK_FP_OPTIMIZED

        with runner.infer_context(context_type) as ctx:
            
            self._hailo_model = runner.get_keras_model(ctx)
            
            if self._hailo_model is None:
                raise ValueError("El modelo Hailo no se cargó correctamente")
            
    def forward(self, data_dict):        
        spatial_features = data_dict['spatial_features']
        
        print('spatial_features.shape', spatial_features.shape)
        spatial_features_hailoinp = np.transpose(spatial_features.cpu().detach().numpy(), (0,2,3,1))
        
        # spatial_features_hailoinp = spatial_features.cpu().detach().numpy()
        print('spatial_features_hailoinp.shape', spatial_features_hailoinp.shape)
        # ============ Hailo-emulation of the Hailo-mapped part ==========
        # spatial_features_2d, box_preds, dir_cls_preds, box_preds_2d, cls_preds  = \
        #                     self._hailo_model(spatial_features_hailoinp)
        spatial_features_2d, cls_preds, box_preds, dir_cls_preds = \
                            self._hailo_model(spatial_features_hailoinp)
        # ================================================================
        print('goo')
        print(cls_preds.shape, type(cls_preds), box_preds.shape, dir_cls_preds.shape)
        cls_preds = torch.Tensor(cls_preds.numpy()) # .permute(0, 2, 3, 1).contiguous()          # [N, H, W, C]
        box_preds = torch.Tensor(box_preds.numpy()) # .permute(0, 2, 3, 1).contiguous()          # [N, H, W, C]
        dir_cls_preds = torch.Tensor(dir_cls_preds.numpy()) # .permute(0, 2, 3, 1).contiguous()
                
        data_dict['spatial_features_2d'] = torch.Tensor(spatial_features_2d.numpy())
        
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=data_dict['batch_size'],
            cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
        )
        data_dict['batch_cls_preds'] = batch_cls_preds
        data_dict['batch_box_preds'] = batch_box_preds
        data_dict['cls_preds_normalized'] = False

        return data_dict
    
def quick_test(runner, hailoize=True, emulate_quantized=False, use_hw=False, emulate_native=False, emulate_optimized=False, fname='/local/shared_with_docker/PointPillars/data/kitti', verbose=False):
    """ Encapsulates a minimalistic test of the complete network with/without hailo offload emulation 
    """
    demo_dataset = ohu.DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(fname), ext=pc_file_extention, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    model_h = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model_h.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
    model_h.eval()

    
    #bb2d_hailo1 = BB2d_Hailo(runner, pppost_onnx='./pp_tmp_post.onnx', emulate_quantized=emulate_quantized, use_hw=use_hw)
    bev_w_head_hailo = Bev_W_Head_Hailo(runner, generate_predicted_boxes=model_h.dense_head.generate_predicted_boxes,
                                        emulate_quantized=emulate_quantized, use_hw=use_hw,  emulate_optimized=emulate_optimized, emulate_native=emulate_native)    
    
    
    # print('################# MODEL BEFORE HAILOIZE #################')
    # print(model_h.module_list)



    if hailoize:        
        # ==== Hook a call into Hailo by replacing parts of sequence by our rigged submodule ====
        model_h.module_list = model_h.module_list[:2] + [bev_w_head_hailo]
        
        
        # print('################# MODEL AFTER HAILOIZE #################')
        # print(model_h.module_list)

        # =======================================================================================                                          
    
    model_cpu = ohu.PointPillarsCPU(model_h) 
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    for idx, data_dict in enumerate(demo_dataset):
        if idx >= 1:
            break  # Solo procesar los primeros 10
        logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])            
        # pred_dicts, _ = model_cpu.forward(data_dict)
        pred_dicts = model_cpu.forward(data_dict)
        if verbose:
            print(pred_dicts)
        else:
            print(f"Numero de pred_scores: {len(pred_dicts[0][0]['pred_scores'])}")
            print(f"Pred_scores (7): {pred_dicts[0][0]['pred_scores'][:20]}")

            print(pred_dicts[0][0].keys())

        
        # ----- Aquí añadimos la parte de visualización -----
        # 1) Averigua el path de la nube
        sample_file = demo_dataset.sample_file_list[idx]  # p.ej. "000123.bin" o ".npy"
        cloud_path = os.path.join(demo_pointcloud, sample_file)

        # 2) Cárgala en NumPy
        points = V.load_point_cloud(cloud_path)

        # 3) Extrae las predicciones
        pred = pred_dicts[0][0]
        boxes  = pred['pred_boxes'].cpu().detach().numpy()
        scores = pred['pred_scores'].cpu().detach().numpy()
        labels = pred['pred_labels'].cpu().detach().numpy()

        print(f"Frame {idx} → {len(boxes)} cajas")

        # 4) Dibuja
        V.draw_scenes(
            points=points,
            ref_boxes=boxes,
            ref_scores=scores,
            ref_labels=labels
        )
    



quick_test(runner, hailoize=False, emulate_quantized=False, fname=demo_pointcloud, emulate_native=False,emulate_optimized=False)     
quick_test(runner, hailoize=True, emulate_quantized=False, fname=demo_pointcloud, emulate_native=False,emulate_optimized=False)     


# # This should give exact same result as we're yet to actually emulate the HW datapath,
# # with its "lossy-compression" (e.g., 8b) features. 
# # This will be possible after calibration and quantization of the model which will also enabling compilation for a physical HW.


    
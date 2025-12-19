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
from pcdet.models import build_network, load_data_to_gpu


import onnx


from hailo_sdk_client import ClientRunner
import hailo_sdk_client
print(hailo_sdk_client.__version__)



# Import custom utilities for OpenPCDet and Hailo integration
import openpcdet2hailo_utils as ohu;
import open3d_vis_utils as V

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

################################# PATHS and config #################################
model = 'centerpoint-pillar' #pointpillars, centerpoint-pillar
dataset = 'kitti' #kitti, waymo 
logger.info(f'MODEL: {model}, DATASET: {dataset}')

# Paths to model configuration and weights
cfg_file = os.path.join(project_root, 'cfgs', f'{model}_{dataset}.yaml')
ckpt_file = os.path.join(project_root, 'checkpoints', f'{model}_{dataset}.pth')

# Path to output directory for the saved model structure
output_dir = os.path.join(project_root, 'output', model, dataset)
os.makedirs(output_dir, exist_ok=True)

# Path to point cloud data
if dataset == 'kitti':
    pointclouds_dir = os.path.join(project_root, 'data', dataset, 'training', 'velodyne')
    pc_file_extension = '.bin'
else:
    pointclouds_dir = os.path.join(project_root, 'data', dataset)
    pc_file_extension = '.npy'


# Hardware architecture
hw_arch='hailo8'

# ONNX model filenames
onnx_name = os.path.join(output_dir, f'{model}_{dataset}.onnx')
onnx_name_simp = os.path.join(output_dir, f'{model}_{dataset}_simplified.onnx')
har_name = os.path.join(output_dir, f'{model}_{dataset}.har')




end_node_names = ['model/concat1', 'model/conv19', 'model/conv18', 'model/conv20']


# Cargar el modelo ONNX simplificado
model = onnx.load(onnx_name_simp)

runner = ClientRunner(hw_arch=hw_arch)

hn, npz = runner.translate_onnx_model(onnx_name_simp, end_node_names = end_node_names)

#print(f'hn: {hn}')


# Save the translated model
runner.save_har(har_name)
# runner = ClientRunner(har=har_name)


# # # Aquí forzamos la carga manual del modelo
# runner._sdk_backend.update_fp_model(runner._sdk_backend.model)
# print("fp_model is None:", runner._sdk_backend.fp_model is None)


# class Bev_W_Head_Hailo(torch.nn.Module):
#     """ Drop-in replacement to the sequence of original "backbone-2d" and "dense_head" modules, accepting and returning dictionary,
#         while under the hood using Hailo [emulator] implementation for the 2D CNN part, accepting and returning tensors I/O
#     """
    
#     def __init__(self, runner, emulate_quantized=False, use_hw=False, emulate_optimized=False, generate_predicted_boxes=None, emulate_native=False):
#         super().__init__()
#         self._runner = runner
#         self.generate_predicted_boxes = generate_predicted_boxes
        
#         if use_hw:
#             context_type = InferenceContext.SDK_HAILO_HW
#         elif emulate_quantized:
#             context_type = InferenceContext.SDK_QUANTIZED 
#         elif emulate_native:
#             context_type = InferenceContext.SDK_NATIVE
#         else:
#             context_type = InferenceContext.SDK_FP_OPTIMIZED

#         with runner.infer_context(context_type) as ctx:
            
#             self._hailo_model = runner.get_keras_model(ctx)
            
#             if self._hailo_model is None:
#                 raise ValueError("El modelo Hailo no se cargó correctamente")
            
#     def forward(self, data_dict):        
#         spatial_features = data_dict['spatial_features']
        
#         print('spatial_features.shape', spatial_features.shape)
#         spatial_features_hailoinp = np.transpose(spatial_features.cpu().detach().numpy(), (0,2,3,1))
        
#         # spatial_features_hailoinp = spatial_features.cpu().detach().numpy()
#         print('spatial_features_hailoinp.shape', spatial_features_hailoinp.shape)
#         # ============ Hailo-emulation of the Hailo-mapped part ==========
#         # spatial_features_2d, box_preds, dir_cls_preds, box_preds_2d, cls_preds  = \
#         #                     self._hailo_model(spatial_features_hailoinp)
#         spatial_features_2d, cls_preds, box_preds, dir_cls_preds = \
#                             self._hailo_model(spatial_features_hailoinp)
#         # ================================================================
#         print('goo')
#         print(cls_preds.shape, type(cls_preds), box_preds.shape, dir_cls_preds.shape)
#         cls_preds = torch.Tensor(cls_preds.numpy()) # .permute(0, 2, 3, 1).contiguous()          # [N, H, W, C]
#         box_preds = torch.Tensor(box_preds.numpy()) # .permute(0, 2, 3, 1).contiguous()          # [N, H, W, C]
#         dir_cls_preds = torch.Tensor(dir_cls_preds.numpy()) # .permute(0, 2, 3, 1).contiguous()
                
#         data_dict['spatial_features_2d'] = torch.Tensor(spatial_features_2d.numpy())
        
#         batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
#             batch_size=data_dict['batch_size'],
#             cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
#         )
#         data_dict['batch_cls_preds'] = batch_cls_preds
#         data_dict['batch_box_preds'] = batch_box_preds
#         data_dict['cls_preds_normalized'] = False

#         return data_dict
    
# def quick_test(runner, hailoize=True, emulate_quantized=False, use_hw=False, emulate_native=False, emulate_optimized=False, fname='/local/shared_with_docker/PointPillars/data/kitti', verbose=False):
#     """ Encapsulates a minimalistic test of the complete network with/without hailo offload emulation 
#     """
#     demo_dataset = ohu.DemoDataset(
#         dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
#         root_path=Path(fname), ext=pc_file_extension, logger=logger
#     )
#     logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
#     model_h = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
#     model_h.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=True)
#     model_h.eval()

    
#     #bb2d_hailo1 = BB2d_Hailo(runner, pppost_onnx='./pp_tmp_post.onnx', emulate_quantized=emulate_quantized, use_hw=use_hw)
#     bev_w_head_hailo = Bev_W_Head_Hailo(runner, generate_predicted_boxes=model_h.dense_head.generate_predicted_boxes,
#                                         emulate_quantized=emulate_quantized, use_hw=use_hw,  emulate_optimized=emulate_optimized, emulate_native=emulate_native)    
    
    
#     # print('################# MODEL BEFORE HAILOIZE #################')
#     # print(model_h.module_list)



#     if hailoize:        
#         # ==== Hook a call into Hailo by replacing parts of sequence by our rigged submodule ====
#         model_h.module_list = model_h.module_list[:2] + [bev_w_head_hailo]
        
        
#         # print('################# MODEL AFTER HAILOIZE #################')
#         # print(model_h.module_list)

#         # =======================================================================================                                          
    
#     model_cpu = ohu.PointPillarsCPU(model_h) 
#     logger.info(f'Total number of samples: \t{len(demo_dataset)}')

#     for idx, data_dict in enumerate(demo_dataset):
#         if idx >= 1:
#             break  # Solo procesar los primeros 10
#         logger.info(f'Visualized sample index: \t{idx + 1}')
#         data_dict = demo_dataset.collate_batch([data_dict])            
#         # pred_dicts, _ = model_cpu.forward(data_dict)
#         pred_dicts = model_cpu.forward(data_dict)
#         if verbose:
#             print(pred_dicts)
#         else:
#             print(f"Numero de pred_scores: {len(pred_dicts[0][0]['pred_scores'])}")
#             print(f"Pred_scores (7): {pred_dicts[0][0]['pred_scores'][:20]}")

#             print(pred_dicts[0][0].keys())

        
#         # ----- Aquí añadimos la parte de visualización -----
#         # 1) Averigua el path de la nube
#         sample_file = demo_dataset.sample_file_list[idx]  # p.ej. "000123.bin" o ".npy"
#         cloud_path = os.path.join(pointclouds_dir, sample_file)

#         # 2) Cárgala en NumPy
#         points = V.load_point_cloud(cloud_path)

#         # 3) Extrae las predicciones
#         pred = pred_dicts[0][0]
#         boxes  = pred['pred_boxes'].cpu().detach().numpy()
#         scores = pred['pred_scores'].cpu().detach().numpy()
#         labels = pred['pred_labels'].cpu().detach().numpy()

#         print(f"Frame {idx} → {len(boxes)} cajas")

#         # 4) Dibuja
#         V.draw_scenes(
#             points=points,
#             ref_boxes=boxes,
#             ref_scores=scores,
#             ref_labels=labels
#         )
    



# quick_test(runner, hailoize=False, emulate_quantized=False, fname=pointclouds_dir, emulate_native=False,emulate_optimized=False)     
# quick_test(runner, hailoize=True, emulate_quantized=False, fname=pointclouds_dir, emulate_native=False,emulate_optimized=False)     


# # # This should give exact same result as we're yet to actually emulate the HW datapath,
# # # with its "lossy-compression" (e.g., 8b) features. 
# # # This will be possible after calibration and quantization of the model which will also enabling compilation for a physical HW.


    
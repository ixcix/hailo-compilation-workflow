
### Import Libraries and Set Paths ###

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Replace with your OpenPCDet clone directory
open3d_ml_clonedir = '/local/shared_with_docker/PointPillars/src/Open3D-ML'
# sys.path.append(openpcdet_clonedir)
sys.path.append(open3d_ml_clonedir + '/tools/')

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



class PointPillars_full(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vfe = model.module_list[0]
        self.scatter = model.module_list[1]
        self.backbone = model.module_list[2]
        self.dense = model.module_list[3] 
        print(self.vfe, self.scatter, self.backbone, self.dense)

    def forward(self, voxels, voxel_num_points, voxel_coords):
        # ===== Paso 1: crear batch_dict inicial =====
        batch_dict = {
            'voxels': voxels,  
            'voxel_num_points': voxel_num_points, 
            'voxel_coords': voxel_coords, 
            'batch_size': 1
        }

        # ===== Paso 2: VFE =====
        batch_dict = self.vfe(batch_dict)

        # ===== Paso 3: Scatter =====
        batch_dict = self.scatter(batch_dict)
        spatial_features = batch_dict['spatial_features']

        # # ===== Paso 4: Backbone =====
        # batch_dict = self.backbone(batch_dict)
        # spatial_features_2d = batch_dict['spatial_features_2d']

        # # ===== Paso 5: Dense Head =====
        # cls_preds = self.dense.conv_cls(spatial_features_2d)
        # box_preds = self.dense.conv_box(spatial_features_2d)
        # dir_cls_preds = self.dense.conv_dir_cls(spatial_features_2d)

        # # ===== Paso 6: devolver las 4 salidas =====
        # return (spatial_features_2d, cls_preds, box_preds, dir_cls_preds)

        # ===== Paso 4: Backbone =====
        ups = []
        ret_dict = {}
        x = spatial_features
        # print('x.shape', x.shape)
        for i in range(len(self.backbone.blocks)):
            x = self.backbone.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.backbone.deblocks) > 0:
                ups.append(self.backbone.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.backbone.deblocks) > len(self.backbone.blocks):
            x = self.backbone.deblocks[-1](x)
        
        spatial_features_2d = x
        
        # ===== Paso 5: Dense Head =====
        cls_preds = self.dense.conv_cls(spatial_features_2d)
        box_preds = self.dense.conv_box(spatial_features_2d)
        dir_cls_preds = self.dense.conv_dir_cls(spatial_features_2d)
        
        return (spatial_features_2d, cls_preds, box_preds, dir_cls_preds)


################################# PATHS and config #################################
model = 'pointpillars'
dataset = 'kitti' #kitti, waymo or innovizone
print(f'DATASET: {dataset}')

# Paths to model configuration and weights
yaml_name = f'/local/shared_with_docker/PointPillars/cfgs/{model}_{dataset}.yaml'
pth_name = f'/local/shared_with_docker/PointPillars/model/{model}_{dataset}.pth'
yaml_name_orig = f'/local/shared_with_docker/PointPillars/cfgs/{model}_{dataset}_orig.yaml'

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

# demo_dataset = ohu.DemoDataset(
#     dataset_cfg=cfg.DATA_CONFIG,
#     class_names=cfg.CLASS_NAMES,
#     training=False,
#     root_path=Path(sample_pointclouds),
#     ext=pc_file_extention,
#     logger=logger
# )

# logger.info(f'Total number of samples: \t{len(demo_dataset)}')

# # Build the model and load pretrained weights
# model = build_network(
#     model_cfg=cfg.MODEL,
#     num_class=len(cfg.CLASS_NAMES),
#     dataset=demo_dataset
# )
# model.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
# model.eval()


################################# Run a Sanity Test #################################

def get_model(cfg, pth_name, demo_dataset):    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
    model.eval()
    return model

# def cfg_from_yaml_file_wrap(yaml_name, cfg):
#     cwd = os.getcwd()
#     os.chdir(openpcdet_clonedir+'/tools/')
#     cfg_from_yaml_file(yaml_name, cfg)
#     os.chdir(cwd)


# cfg_from_yaml_file_wrap(yaml_name, cfg)



logger.info('------ Quick Demo of OpenPCDet''s PointPillarsCPU -------')

cfg_from_yaml_file(yaml_name_orig, cfg)

demo_dataset = ohu.DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    root_path=Path(demo_pointcloud), ext=pc_file_extention, logger=logger
)
logger.info(f'Total number of samples: \t{len(demo_dataset)}')

model = get_model(cfg, pth_name, demo_dataset)
model.eval()
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
# We will offload the model COMPLETE to the Hailo device.

cfg_from_yaml_file(yaml_name, cfg)
model = get_model(cfg, pth_name, demo_dataset)
model.eval()
#model.cpu()

model_wrapper = PointPillars_full(model)


# --- 3. Crear Tensores de Entrada (Dummy Inputs) ---
#    (Asegúrate de que 'data_dict' es el que se cargó en la 'Sanity Test' [Línea 216])

print(data_dict.keys())

## crea dummies explícitos a partir de data_dict pero garantizando requires_grad = False
dummy_voxels = data_dict['voxels'].unsqueeze(0)
dummy_voxel_num_points = data_dict['voxel_num_points'].unsqueeze(0)
dummy_voxel_coords = data_dict['voxel_coords'].unsqueeze(0)

print('=== Dummy Inputs ===')
print(f'dummy_voxels {dummy_voxels.shape}')
print(f'dummy_voxel_num_points {dummy_voxel_num_points.shape}')
print(f'dummy_voxel_coords {dummy_voxel_coords.shape}')
print('====================')

# with torch.no_grad():
#     output = model_wrapper(dummy_voxels, dummy_voxel_num_points, dummy_voxel_coords)
# print([x.shape for x in output])


# Definimos los nombres de entrada y salida
input_names = ['voxels', 'voxel_num_points', 'voxel_coords']

# Y luego usa esos nombres aquí:
output_names = ['model/concat1', 'model/conv19', 'model/conv18', 'model/conv20']

# --- 4. Exportar a ONNX ---
print(f"Exportando modelo completo a ONNX con entradas: {input_names}")



with torch.no_grad():
    torch.onnx.export(
        model_wrapper,
        args=(dummy_voxels, dummy_voxel_num_points, dummy_voxel_coords),
        f=onnx_name,
        input_names=['voxels', 'voxel_num_points', 'voxel_coords'],
        output_names=output_names,
        dynamic_axes=None,
        opset_version=13,
        verbose=False
    )


print("Exportación a ONNX completada.")



# Simplify the ONNX model:
model_onnx = onnx.load(onnx_name)
model_simplified, check = simplify(model_onnx)
if check:
    onnx.save(model_simplified, f"{onnx_name_simp}")
    print(f"Modelo simplificado guardado en {onnx_name_simp}")
else:
    print("La simplificación del modelo falló.")


onnx_model = onnx.load(onnx_name_simp)
print("== OUTPUTS ==")
for o in onnx_model.graph.output:
    print(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])

print("== INPUTS ==")
for i in onnx_model.graph.input:
    print(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])    

#exit(0)



# --- 6. Compilar con Hailo SDK ClientRunner ---
runner = ClientRunner(hw_arch=hw_arch)


hn, npz = runner.translate_onnx_model(onnx_name_simp, end_node_names = output_names)
runner.save_har(har_name)
print(f"Modelo .har guardado en: {har_name}")

exit(0)












logger.info('------ Quick Demo del Modelo Hailo (Emulado) -------')

# 1. Obtenemos los tensores de entrada del 'data_dict' (del Sanity Test anterior)
hailo_inputs = {
    'voxels': data_dict['voxels'].cpu().numpy(),
    'voxel_num_points': data_dict['voxel_num_points'].cpu().numpy(),
    'voxel_coords': data_dict['voxel_coords'].cpu().numpy()
}

# 2. Ejecutamos la inferencia en el emulador (SDK_QUANTIZED)
#    Esto simula la ejecución en el Hailo-8 (RPi5)
try:
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        hailo_model = runner.get_keras_model(ctx)
        
        # Preparamos los datos en el formato que espera el .har
        # (El runner de Hailo gestiona el 'batch_dict' interno)
        hailo_results = hailo_model.predict(hailo_inputs)
        
        # 'hailo_results' es un diccionario {nombre_salida: array_numpy}
        # Ejemplo: {'model/concat1': array, 'model/conv19': array, ...}
        
except Exception as e:
    print(f"Error durante la emulación de Hailo: {e}")
    exit(1)


# 3. Ejecutar el Post-procesamiento (NMS) en la CPU
#    (Replicamos lo que hace 'postproc_consumer' en tu RPi5)

# Convertimos los arrays de NumPy (salida de Hailo) a Tensores de PyTorch
# (Añadimos el 'batch_dim' que espera el NMS)
hailo_out_torch = {
    'model/concat1': torch.from_numpy(np.expand_dims(hailo_results['model/concat1'], 0)),
    'model/conv19': torch.from_numpy(np.expand_dims(hailo_results['model/conv19'], 0)),
    'model/conv18': torch.from_numpy(np.expand_dims(hailo_results['model/conv18'], 0)),
    'model/conv20': torch.from_numpy(np.expand_dims(hailo_results['model/conv20'], 0))
}

# Preparamos el 'batch_dict' para la función de post-procesamiento
# (pp_post_bev_w_head es 'ohu.PP_Post_Bev_w_Head(model)' de tu script RPi5)
#
# Necesitamos instanciar 'pp_post_bev_w_head' aquí
pp_post_bev_w_head = ohu.PP_Post_Bev_w_Head(model)

bev_out_list = [
    hailo_out_torch['model/concat1'],
    hailo_out_torch['model/conv19'],
    hailo_out_torch['model/conv18'],
    hailo_out_torch['model/conv20']
]

# Ejecutamos el NMS
pred_dicts = pp_post_bev_w_head(bev_out_list)

# 4. Visualizar los resultados (igual que en el Sanity Test)
print(f"Numero de pred_scores (Hailo): {len(pred_dicts[0]['pred_scores'])}")
print(f"Pred_scores (Hailo) (7): {pred_dicts[0]['pred_scores'][:20]}")

# (El resto del código de visualización)
sample_file = demo_dataset.sample_file_list[idx]
cloud_path = os.path.join(demo_pointcloud, sample_file)
points = V.load_point_cloud(cloud_path)

pred = pred_dicts[0]
boxes  = pred['pred_boxes'].cpu().detach().numpy()
scores = pred['pred_scores'].cpu().detach().numpy()
labels = pred['pred_labels'].cpu().detach().numpy()

print(f"Frame {idx} (Hailo) → {len(boxes)} cajas")

# 4) Dibuja
V.draw_scenes(
    points=points,
    ref_boxes=boxes,
    ref_scores=scores,
    ref_labels=labels
)
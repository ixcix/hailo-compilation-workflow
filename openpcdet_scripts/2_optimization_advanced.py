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

# # Añadir explícitamente la ruta del paquete DALI
# dali_path = "/local/workspace/hailo_virtualenv/lib/python3.10/site-packages"
# if dali_path not in sys.path:
#     sys.path.append(dali_path)
# os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + dali_path



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
log_file = os.path.join(log_dir, 'pp_opt.log')
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

cs_size = 64  # Calibration set size
finetune_size = 256  # Finetuning dataset size
adaraound_size = 256  # Adaround dataset size

#probar tambien con finetune a 64 y 20 epochs y yastaria que me aburro de esperar
#leer doc mix_precision_search aplicado al modelo ya cuantizado.




all_lines_kitti_final_fast3_mod= [   #results fatal
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    # --- Pre-cuánticas ---
    'pre_quantization_optimization(equalization, policy=enabled)',
    'pre_quantization_optimization(dead_layers_removal, policy=enabled)',
    #'pre_quantization_optimization(weights_clipping, layers={conv*}, mode=mmse)',
#quitar conv6,12,12,15
    'pre_quantization_optimization(activation_clipping, layers=[conv1, conv2,conv3,conv4,conv5,conv6,conv7,conv8,conv9,conv10,conv11], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv12, conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[deconv1, deconv2, conv17, conv18, conv19, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1, shuffle=False)'
]



all_lines_kitti_prueba= [    
    'model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'pre_quantization_optimization(equalization, layers={conv*}, policy=enabled)',
    'pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=15, dataset_size={cs_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, policy=enabled, dataset_size={adaraound_size}, batch_size=1)'
]
all_lines_kitti_prueba2= [   #resultados: buenos, algo peor en ped pero el resto guay ademas el f1 score bastante
    'model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    'pre_quantization_optimization(equalization, policy=enabled)',
    'pre_quantization_optimization(dead_layers_removal, policy=enabled)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1)'
]

all_lines_kitti_prueba3= [   #doing: ver resultados, hailo profiler regulero
    'model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    'pre_quantization_optimization(equalization, policy=enabled)',
    'pre_quantization_optimization(dead_layers_removal, policy=enabled)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv5,conv6,conv7,conv8,conv9,conv10,deconv1,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv11,conv12,conv13,conv14,conv15,conv16,conv17], mode=percentile, clipping_values=[0.01, 99.99])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.995])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1)'
]

all_lines_kitti_prueba4= [   #doing: resultados
    'model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    'pre_quantization_optimization(equalization, policy=enabled)',
    'pre_quantization_optimization(dead_layers_removal, policy=enabled)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.99])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1)'
]

all_lines_kitti_prueba5= [   #doing, results: fatal
    'model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    'pre_quantization_optimization(equalization, policy=enabled)',
    'pre_quantization_optimization(dead_layers_removal, policy=enabled)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv17,deconv1,deconv2], mode=percentile, clipping_values=[0.01, 99.99])',
    'pre_quantization_optimization(activation_clipping, layers=[conv11,conv12,conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv19, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1)'
]


all_lines_kitti_prueba6= [   #doing: mal
    'model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    'pre_quantization_optimization(equalization, policy=enabled)',
    'pre_quantization_optimization(dead_layers_removal, policy=enabled)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.99])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv19, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1)'
]

all_lines_kitti_prueba7= [   #doing: mejor 
    'model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    'pre_quantization_optimization(equalization, policy=enabled)',
    'pre_quantization_optimization(dead_layers_removal, policy=enabled)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.99])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    #'pre_quantization_optimization(activation_clipping, layers=[conv18, conv19, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1)'
]

all_lines_kitti_prueba8= [    #basado en all_lines_kitti_2_clipp_manual3_finetune4 pero con las ultimas capas sin clipping y mas finetune y adaround
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    #'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1, shuffle=False)'
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
      
    all_lines = all_lines_waymo
    custom_optimization = True
    nms_hailo = False

elif dataset == 'kitti':

    all_lines = all_lines_kitti_prueba8
    custom_optimization = True 
    nms_hailo = False


elif dataset == 'innovizone':
 
    all_lines = all_lines_innovizone
    custom_optimization = True 
    nms_hailo = False


name = "all_lines_kitti_prueba8"
# Output file names
har_name = f'{output_path}/pp_bev_w_head.har'
opt_har_name = f'{output_path}/pp_bev_w_head_opt_{name}.har'
q_har_name = f'{output_path}/pp_bev_w_head_{name}.q.har'
nms_config_file = '/local/shared_with_docker/PointPillars/src/pointpillars_nms_postprocess_config.json'
#q_har_path = '/local/shared_with_docker/PointPillars/output/pointpillars/kitti/pp_bev_w_head_all_lines_kitti_2_clipp_manual3_finetune3.q.har'


if nms_hailo:
    all_lines.append(f'nms_postprocess("{nms_config_file}", meta_arch=ssd, engine=nn_core)')



try:
    del model, demo_dataset
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
except:
    pass


runner = ClientRunner(har=har_name )


if custom_optimization:
    runner.load_model_script('\n'.join(all_lines))

runner.optimize_full_precision(optimization_dataset)

runner.save_har(opt_har_name)

# Optimize the model and quant
runner.optimize(optimization_dataset)
# runner.analyze_noise(calib_set)
runner.save_har(q_har_name)



torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()
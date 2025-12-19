import os
import sys
import torch
import numpy as np
from pathlib import Path


from hailo_sdk_client import ClientRunner
from hailo_sdk_client import InferenceContext #SdkPartialNumeric, SdkNative # 
import tensorflow as tf
import hailo_sdk_client
print(hailo_sdk_client.__version__)

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

import openpcdet2hailo_utils as ohu;

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network

from hailo_sdk_client import ClientRunner


# Replace with your OpenPCDet clone directory
openpcdet_clonedir = '/local/shared_with_docker/PointPillars/src/OpenPCDet'
sys.path.append(openpcdet_clonedir + '/tools/')


################################# PATHS and config #################################
model = 'pointpillars'
dataset = 'kitti' #kitti, waymo or innovizone

# Paths to model configuration and weights
yaml_name = f'/local/shared_with_docker/PointPillars/cfgs/{model}_{dataset}_orig.yaml'
pth_name = f'/local/shared_with_docker/PointPillars/model/{model}_{dataset}.pth'

# Path to point cloud data custom
pointclouds = f'/local/shared_with_docker/PointPillars/data/{dataset}/training/velodyne/'
imageset = 'val'  # 'val' or 'test' para quick test
imageset_file = f'/local/shared_with_docker/PointPillars/data/{dataset}/ImageSets/{imageset}.txt'
n_demo = 10  # Number of demo samples to run

# File extension of point cloud files
if dataset == 'kitti':
    pc_file_extention = '.bin'
else:
    pc_file_extention = '.npy'

# Path to output dir for the onnx file
output_path = f'/local/shared_with_docker/PointPillars/output/{model}/{dataset}'

cs_size = 64  # Calibration set size

# Output file names
name = "all_lines_kitti_prueba8"
#q_har_name = f'{output_path}/pp_bev_w_head_{cs_size}.q.har'
q_har_name = f'{output_path}/pp_bev_w_head_{name}.q.har'
#hef_name = f'{output_path}/pp_bev_w_head_{cs_size}.hef'
hef_name = f'{output_path}/pp_bev_w_head_{name}.hef'


# Hardware architecture
hw_arch='hailo8'

# Directory where logs should be saved
log_dir = '/local/shared_with_docker/PointPillars/logs'
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
log_file = os.path.join(log_dir, 'pp_comp.log')
logger = common_utils.create_logger(log_file)



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
# demo_dataset = ohu.DemoDataset(
#     dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
#     root_path=Path(pointclouds), ext=pc_file_extention, logger=logger
# )
# # logger.info(f'Total number of samples: \t{len(demo_dataset)}')

# model = get_model(cfg, pth_name, demo_dataset)
# model.cuda()
# model.eval()

# np.set_printoptions(precision=2)



# # Load quantized model
# runner = ClientRunner(har=q_har_name)

# # Compile the model
# runner.load_model_script('context_switch_param(mode=disabled)')
# compiled_model = runner.compile()

# # Save the compiled model
# with open(hef_name, 'wb') as f:
#     f.write(compiled_model)


class Bev_W_Head_Hailo(torch.nn.Module):
    """ Drop-in replacement to the sequence of original "backbone-2d" and "dense_head" modules, accepting and returning dictionary,
        while under the hood using Hailo [emulator] implementation for the 2D CNN part, accepting and returning tensors I/O
    """
    
    def __init__(self, runner, emulate_quantized=False, use_hw=False, emulate_optimized= False, generate_predicted_boxes=None):
        super().__init__()
        self._runner = runner
        self.generate_predicted_boxes = generate_predicted_boxes
        
        if use_hw:
            context_type = InferenceContext.SDK_HAILO_HW
        elif emulate_quantized:
            context_type = InferenceContext.SDK_QUANTIZED 
        elif emulate_optimized:
            context_type = InferenceContext.SDK_FP_OPTIMIZED
        else:
            context_type = InferenceContext.SDK_NATIVE
            
        with runner.infer_context(context_type) as ctx:
            
            self._hailo_model = runner.get_keras_model(ctx)
            
            if self._hailo_model is None:
                raise ValueError("El modelo Hailo no se cargó correctamente")
            
    def forward(self, data_dict):        
        spatial_features = data_dict['spatial_features']
        
        spatial_features_hailoinp = np.transpose(spatial_features.cpu().detach().numpy(), (0,2,3,1))
        
        # ============ Hailo-emulation of the Hailo-mapped part ==========
        spatial_features_2d, cls_preds, box_preds, dir_cls_preds = \
                            self._hailo_model(spatial_features_hailoinp)
        # ================================================================
        
        print(cls_preds.shape, type(cls_preds), box_preds.shape)
        cls_preds = torch.Tensor(cls_preds.numpy())                                         # .permute(0, 2, 3, 1).contiguous()           [N, H, W, C]
        box_preds = torch.Tensor(box_preds.numpy())                                         #.permute(0, 2, 3, 1).contiguous()           [N, H, W, C]
        dir_cls_preds = torch.Tensor(dir_cls_preds.numpy()).permute(0, 2, 3, 1).contiguous()
                
        data_dict['spatial_features_2d'] = torch.Tensor(spatial_features_2d.numpy())
        
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=data_dict['batch_size'],
            cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
        )
        data_dict['batch_cls_preds'] = batch_cls_preds
        data_dict['batch_box_preds'] = batch_box_preds
        data_dict['cls_preds_normalized'] = False

        return data_dict

def quick_test(runner, hailoize=True, emulate_quantized=False, use_hw=False, fnames='/local/shared_with_docker/PointPillars/data/kitti/training/velodyne/000001.bin', verbose=False):
    """ Encapsulates a minimalistic test of the complete network with/without hailo offload emulation 
    """
    if hailoize:
        logger.info("Starting quick test with hailoize=True and emulate_quantized={}".format(emulate_quantized))
    else:
        logger.info("Starting quick test with hailoize=False")


    demo_dataset = ohu.DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(pointclouds), ext=pc_file_extention, logger=logger
    )
    #logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    model_h = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model_h.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
    model_h.eval()

    
    #bb2d_hailo1 = BB2d_Hailo(runner, pppost_onnx='./pp_tmp_post.onnx', emulate_quantized=emulate_quantized, use_hw=use_hw)
    bev_w_head_hailo = Bev_W_Head_Hailo(runner, generate_predicted_boxes=model_h.dense_head.generate_predicted_boxes,
                                        emulate_quantized=emulate_quantized, use_hw=use_hw)    
    
    
    if hailoize:        
        # ==== Hook a call into Hailo by replacing parts of sequence by our rigged submodule ====
        model_h.module_list = model_h.module_list[:2] + [bev_w_head_hailo]
        # =======================================================================================                                          
    
    model_cpu = ohu.PointPillarsCPU(model_h) 
    logger.info(f"Processing {len(fnames)} frames for quick test...")

    all_means = []
    for idx, frame_id in enumerate(fnames):
        pc_path = f'/local/shared_with_docker/PointPillars/data/{dataset}/training/velodyne/{frame_id}{pc_file_extention}'
        if not os.path.exists(pc_path):
            logger.warning(f"[WARN] Missing file: {pc_path}")
            continue

        data_dict = demo_dataset.collate_batch([demo_dataset.__getitem__(int(frame_id))])
        pred_dicts = model_cpu.forward(data_dict)
        scores = pred_dicts[0][0]['pred_scores'].detach().cpu().numpy()

        mean_score = scores.mean()
        all_means.append(mean_score)

        if verbose:
            print(f"[{idx+1}/{len(fnames)}] Frame {frame_id}: mean={mean_score:.4f}, top5={scores[:5]}")
        else:
            print(f"[{idx+1}/{len(fnames)}] {frame_id}: {scores[:7]}")

    # --- Estadísticas globales ---
    if all_means:
        logger.info(f"[SUMMARY] mean={np.mean(all_means):.4f}, std={np.std(all_means):.4f}, "
                    f"min={np.min(all_means):.4f}, max={np.max(all_means):.4f}")
    else:
        logger.warning("No frames were processed successfully.")

# Leer los IDs del split
with open(imageset_file, 'r') as f:
    frame_ids = [line.strip() for line in f.readlines()]
frame_ids = frame_ids[:n_demo]  # usa solo los primeros N
print(f"[INFO] Running quick test on {len(frame_ids)} frames from {imageset_file}")



# Load quantized model
runner = ClientRunner(har=q_har_name)

quick_test(runner, hailoize=False, fnames = frame_ids)

quick_test(runner, hailoize=True, emulate_quantized=True, fnames = frame_ids)



do_compile = True
if do_compile:
    alls_line = ['shortcut_concat1_conv20 = shortcut(concat1, conv20)',
                 'performance_param(compiler_optimization_level=max)']
    #open('helper.alls','w').write(alls_line1)  #   !!!!
    
    
    runner.load_model_script('\n'.join(alls_line))
    #runner.load_model_script('./helper.alls') 

    compiled_model=runner.compile()    
    open(hef_name, 'wb').write(compiled_model)

# Expected results:
# [info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
# [info] +-----------+---------------------+---------------------+--------------------+
#                                        ...
# [info] +-----------+---------------------+---------------------+--------------------+
# [info] | Total     | 62.5%               | 63.7%               | 37.2%              |



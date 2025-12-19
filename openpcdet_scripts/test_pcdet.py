# test_eval_only.py
import pickle
import copy
import numba
import os
import sys
from pathlib import Path
#numba.config.DISABLE_JIT = True
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from pcdet.config import cfg, cfg_from_yaml_file

# --- Import Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)


pkl = "/local/shared_with_docker/hailo-compilation-workflow/results_nms_bev_0.3.pkl"
with open(pkl, "rb") as f:
    preds = pickle.load(f)

# Solo 1 pred y 1 gt
det_annos = [preds[0]]
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

# Configuration: Choose model and dataset
model = 'centerpoint-pillar' # Options: pointpillars, centerpoint-pillar
dataset = 'kitti'      # Options: kitti


# Paths to model configuration and weights
cfg_file = os.path.join(project_root, 'cfgs', f'{model}_{dataset}.yaml')
cfg_from_yaml_file(cfg_file, cfg)

pointclouds_dir = os.path.join(project_root, 'data', dataset, 'training', 'velodyne')
dataset = KittiDataset(root_path=Path(pointclouds_dir), dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False)

# Cargar solo una gt
gt = dataset.kitti_infos[0]['annos']
gt_annos = [gt]

print("Llamando a get_official_eval_result...")
print(det_annos)
print(gt_annos)

kitti_eval.get_official_eval_result(gt_annos, det_annos, ['Car','Pedestrian','Cyclist'])

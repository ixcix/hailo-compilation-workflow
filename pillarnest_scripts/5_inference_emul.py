import os
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, 
                            InputVStreamParams, OutputVSrteamParams, InputVStreams, 
                            OutputVStreams, FormatType)
import sys
import os

# Añadimos la raíz del proyecto al path de Python
# Como lanzas desde /local/shared_with_docker/hailo-compilation-workflow/
# y la carpeta 'model' está ahí mismo, esto lo solucionará:
sys.path.append(os.getcwd())
# --- TUS CLASES ---
from model.hailo8l.pillarnest_retraining.pillarnest_logic_pre import PillarnestLoader, PillarnestMultiSweep, PillarnestVoxelizer, PillarnestHeightEncoder, PillarnestScatter
from model.hailo8l.pillarnest_retraining.pillarnest_logic_post import CenterPointPostProcessor
from model.hailo8l.pillarnest_retraining.pillarnest_config import PillarnestTinyConfig as Cfg

# ==========================================
# CONFIGURACIÓN DE RUTAS Y MODELO
# ==========================================
PKL_INFOS = "/local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_retraining/val_infos.pkl"
DATA_ROOT = "/local/shared_with_docker/hailo-compilation-workflow/data/nuscenes/v1.0-trainval/"
WEIGHTS_NUMPY = "/local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_retraining/pillarnest_encoder_weights.npy"

# --- CONFIGURACIÓN DE PRUEBA ---
LIMIT_FRAMES = 50  # <-- Cambia a None para procesar TODO el dataset (6,019 frames)

# --- SELECCIÓN DE MODELO ---
# MODELO_A_PROBAR = "pillarnest_tiny_hailo_retraining.har"   # FP32
# MODELO_A_PROBAR = "pillarnest_tiny_hailo_retraining.q.har" # Cuantizado
MODELO_A_PROBAR = "pillarnest_tiny_hailo_retraining.hef" # Compilado  

MODEL_PATH = os.path.join("/local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_retraining/", MODELO_A_PROBAR)

# Nombre de salida dinámico
suffix = f"_limit_{LIMIT_FRAMES}" if LIMIT_FRAMES else "_full"
OUTPUT_PKL = MODEL_PATH.replace(".har", f"{suffix}_results.pkl")

def get_pose_matrix(translation, rotation):
    from pyquaternion import Quaternion
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = translation
    return matrix

class PillarnestHailoRunner:
    def __init__(self, model_path, weights_path):
        print(f"🚀 Cargando modelo para emulación: {os.path.basename(model_path)}")
        self.target = VDevice()
        self.hef = HEF(model_path)
        self.configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        
        is_quantized = ".q.har" in model_path or ".hef" in model_path
        self.vstream_info = {"quantized": is_quantized}

        self.loader = PillarnestLoader(load_dim=5)
        self.sweeper = PillarnestMultiSweep(sweeps_num=Cfg.sweeps_num, remove_close=True, test_mode=True)
        self.voxelizer = PillarnestVoxelizer(Cfg.voxel_size, Cfg.point_cloud_range, Cfg.max_num_points, Cfg.max_voxels[1])
        self.encoder = PillarnestHeightEncoder(Cfg.voxel_size, Cfg.point_cloud_range, [Cfg.feat_channels])
        self.encoder.set_weights(np.load(weights_path, allow_pickle=True).item())
        self.scatter = PillarnestScatter(output_shape=Cfg.grid_size[:2], num_input_features=48)
        self.post_processor = CenterPointPostProcessor(Cfg)

    def run(self, val_infos, limit=None):
        results_mmdet = []
        # Aplicar límite si existe
        if limit:
            val_infos = val_infos[:limit]
            print(f"⚠️ Modo PRUEBA activado: Solo se procesarán los primeros {limit} frames.")
        
        with self.network_group.activate(self.network_group_params):
            input_params = InputVStreamParams.make(self.network_group, 
                                                  quantized=self.vstream_info["quantized"], 
                                                  format_type=FormatType.FLOAT32)
            output_params = OutputVStreamParams.make_from_network_group(self.network_group, 
                                                                       quantized=self.vstream_info["quantized"], 
                                                                       format_type=FormatType.FLOAT32)

            with InputVStreams(self.network_group, input_params) as v_in, \
                 OutputVStreams(self.network_group, output_params) as v_out:
                
                in_stream = list(v_in)[0]
                out_streams = list(v_out)
                out_names = [self.hef.get_original_names_from_vstream_name(s.name)[0] for s in out_streams]

                for info in tqdm(val_infos, desc=f"Procesando {MODELO_A_PROBAR}"):
                    # Pre-procesado
                    rel_path = info['lidar_path'].split('./data/nuscenes/')[-1]
                    print(info['lidar_path'])  # Ruta original del .bin
                    print(rel_path)  # Verificar que las rutas sean correctas
                    # exit()
                    info['filename'] = os.path.join(DATA_ROOT, rel_path)
                    print(info['filename'])  # Verificar ruta final del .bin
                    
                    l2e = get_pose_matrix(info['lidar2ego_translation'], info['lidar2ego_rotation'])
                    e2g = get_pose_matrix(info['ego2global_translation'], info['ego2global_rotation'])
                    inv_curr_pose = np.linalg.inv(e2g @ l2e)
                    
                    sweeps_info = []
                    for sw in info['sweeps']:
                        s_copy = sw.copy()
                        p_pose = get_pose_matrix(sw['ego2global_translation'], sw['ego2global_rotation']) @ \
                                 get_pose_matrix(sw['sensor2ego_translation'], sw['sensor2ego_rotation'])
                        rel = inv_curr_pose @ p_pose
                        s_copy['sensor2lidar_rotation'], s_copy['sensor2lidar_translation'] = rel[:3, :3], rel[:3, 3]
                        sweeps_info.append(s_copy)

                    points = self.loader.load_points(info['filename'])
                    res_cust = {'points': points, 'timestamp': info['timestamp'] / 1e6, 'sweeps': sweeps_info}
                    points_multi = self.sweeper.process(res_cust)['points']
                    
                    voxels, coors, n_pts = self.voxelizer.voxelize(points_multi)
                    feat = self.encoder.encode(voxels, n_pts, coors)
                    canvas = self.scatter.scatter(feat, coors)

                    # Inferencia
                    in_stream.send(np.ascontiguousarray(canvas, dtype=np.float32))
                    raw_outputs = [s.recv() for s in out_streams]
                    
                    hailo_dict = {name: out[0].transpose(2,0,1) if out.ndim==4 else out.transpose(2,0,1) 
                                 for name, out in zip(out_names, raw_outputs)}

                    # Post-procesado y Formateo
                    custom_res = self.post_processor.forward(hailo_dict)
                    formatted = self.format_for_mmdet(custom_res)
                    results_mmdet.append({'pts_bbox': formatted})

        return results_mmdet

    def format_for_mmdet(self, res_dicts):
        boxes, scores, labels = [], [], []
        class_to_id = {name: i for i, name in enumerate(Cfg.class_names)}
        for r in res_dicts:
            b = r['box']
            vx, vy = r['velocity']
            boxes.append([b['x'], b['y'], b['z'], b['l'], b['w'], b['h'], b['rot'], vx, vy])
            scores.append(r['score'])
            labels.append(class_to_id.get(r['label'], -1))
        return {
            'boxes_3d': np.array(boxes),
            'scores_3d': np.array(scores),
            'labels_3d': np.array(labels)
        }

if __name__ == "__main__":
    with open(PKL_INFOS, 'rb') as f:
        infos = pickle.load(f)
    
    runner = PillarnestHailoRunner(MODEL_PATH, WEIGHTS_NUMPY)
    final_results = runner.run(infos, limit=LIMIT_FRAMES)
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(final_results, f)
    print(f"✅ Prueba terminada. Resultados guardados en {OUTPUT_PKL}")
import os
import sys
import pickle
import copy
import numpy as np
from tqdm import tqdm

# MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAL

# 🛠️ IMPORTANTE: Importamos InferVStreams para la inferencia síncrona
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType)

sys.path.append("/local/shared_with_docker/hailo-compilation-workflow")

from pillarnest_scripts.pillarnest_logic_pre import (
    PillarnestLoader, PillarnestMultiSweep, PillarnestVoxelizer, 
    PillarnestHeightEncoder, PillarnestScatter
)
from pillarnest_scripts.pillarnest_logic_post import CenterPointPostProcessor
from pillarnest_scripts.pillarnest_config import PillarnestTinyConfig as Cfg

# ==========================================
# CONFIGURACIÓN
# ==========================================
PKL_INFOS = "/local/shared_with_docker/hailo-compilation-workflow/data/nuscenes/v1.0-trainval/nuscenes_infos_val.pkl"
DATA_ROOT = "/local/shared_with_docker/hailo-compilation-workflow/data/nuscenes/v1.0-trainval"
WEIGHTS_NUMPY = "/local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_original/pillarnest_encoder_weights.npy"
MODEL_PATH = "/local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_original/pillarnest_tiny_hailo_opt12.hef"

LIMIT_FRAMES = None 

def fix_path(path, data_root=DATA_ROOT):
    p = path.replace('\\', '/')
    if 'samples' in p:
        rel = p.split('samples/')[-1]
        return os.path.join(data_root, 'samples', rel)
    if 'sweeps' in p:
        rel = p.split('sweeps/')[-1]
        return os.path.join(data_root, 'sweeps', rel)
    return os.path.join(data_root, os.path.basename(p))

class PillarnestHailoRunner:
    def __init__(self, model_path, weights_path):
        print(f"✅ Inicializando HailoRT...")
        self.target = VDevice()
        self.hef = HEF(model_path)
        self.network_group = self.target.configure(self.hef, ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe))[0]
        self.network_group_params = self.network_group.create_params()
        
        self.loader = PillarnestLoader(load_dim=5)
        self.sweeper = PillarnestMultiSweep(sweeps_num=Cfg.sweeps_num, remove_close=True, test_mode=True)
        self.voxelizer = PillarnestVoxelizer(Cfg.voxel_size, Cfg.point_cloud_range, Cfg.max_num_points, Cfg.max_voxels[1])
        self.encoder = PillarnestHeightEncoder(Cfg.voxel_size, Cfg.point_cloud_range, [Cfg.feat_channels])
        self.encoder.set_weights(np.load(weights_path, allow_pickle=True).item())
        self.scatter = PillarnestScatter(output_shape=Cfg.grid_size[:2], num_input_features=48)
        self.post_processor = CenterPointPostProcessor(Cfg)

    def run(self, val_infos, limit=None):
        results_mmdet = []
        val_infos = val_infos[:limit] if limit else val_infos
        
        output_nodes_info = {}

        for info in self.hef.get_output_vstream_infos():
            output_nodes_info[info.name] = {
                "original_names": self.hef.get_original_names_from_vstream_name(info.name),
                "shape": info.shape
            }

        # Extraemos el nombre exacto de la entrada del modelo
        input_vstream_name = self.hef.get_input_vstream_infos()[0].name

        v_in_p = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        v_out_p = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)

        # 🛠️ INFERENCIA SÍNCRONA: Usamos InferVStreams como context manager principal
        with InferVStreams(self.network_group, v_in_p, v_out_p) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):

                for info in tqdm(val_infos, desc="Inferencia Hailo Síncrona"):
                    # --- PRE-PROCESADO ---
                    lidar_f = fix_path(info['lidar_path'])
                    if not os.path.exists(lidar_f): 
                        print(f"⚠️ LiDAR no encontrado: {lidar_f}")
                        continue
                    
                    pts = self.loader.load_points(lidar_f)
                    
                    sweeps_fixed = []
                    for s in info.get('sweeps', []):
                        s_copy = copy.deepcopy(s)
                        s_copy['data_path'] = fix_path(s['data_path'])
                        sweeps_fixed.append(s_copy)

                    pts_m = self.sweeper.process({
                        'points': pts, 
                        'timestamp': info['timestamp']/1e6, 
                        'sweeps': sweeps_fixed,
                        'lidar_path': lidar_f
                    })['points']
                    
                    v, c, n = self.voxelizer.voxelize(pts_m)
                    feat = self.encoder.encode(v, n, c)
                    canvas = self.scatter.scatter(feat, c)

                    # --- INFERENCIA ---
                    # 1. Empaquetar y asegurar memoria contigua
                    input_data = np.expand_dims(canvas, axis=0).astype(np.float32)
                    input_data = np.ascontiguousarray(input_data)
                    
                    # 2. Llamada Bloqueante (Síncrona). Esperamos hasta tener los 42 resultados
                    infer_results = infer_pipeline.infer({input_vstream_name: input_data})
                    
                    # --- EXTRACCIÓN Y TRANSPOSICIÓN ---
                    output_map = {}
                    for stream_name, data in infer_results.items():
                        
                        # Transponer de Hailo (NHWC o HWC) a PyTorch (NCHW) de forma segura
                        if data.ndim == 4: # Viene como (1, H, W, C)
                            data_nchw = np.ascontiguousarray(data.transpose(0, 3, 1, 2))
                        elif data.ndim == 3: # Viene como (H, W, C)
                            data_nchw = np.ascontiguousarray(data.transpose(2, 0, 1)[None, :, :, :])
                        else:
                            print(f"⚠️ Dimensión inesperada en {stream_name}: {data.shape}")
                            sys.exit()

                        # Mapear al nombre original que usa el post-procesador
                        for orig_name in output_nodes_info[stream_name]['original_names']:
                            output_map[orig_name] = data_nchw

                    # --- POST-PROCESADO ---
                    headers = self.post_processor.headers
                    num_tasks = len(self.post_processor.tasks)

                    raw_res = []
                    for task_id in range(num_tasks):
                        for header in headers:
                            key = f"task{task_id}_{header}"
                            if key in output_map:
                                raw_res.append(output_map[key])
                            else:
                                print(f"⚠️ Output {key} no encontrado en HEF outputs!")

                    preds = self.post_processor.forward(raw_res)

                    # --- GUARDADO PARA MMDET3D ---
                    results_mmdet.append({
                        'pts_bbox': self.format_for_mmdet(preds),
                        'token': info.get('token', ''),
                        'sample_idx': info.get('token', '')
                    })
                
            return results_mmdet

    def format_for_mmdet(self, res_dicts):
        boxes, scores, labels = [], [], []
        c2id = {name: i for i, name in enumerate(Cfg.class_names)}
        for r in res_dicts:
            b = r['box']
            boxes.append([b['x'], b['y'], b['z'], b['l'], b['w'], b['h'], b['rot'], r['velocity'][0], r['velocity'][1]])
            scores.append(r['score'])
            labels.append(c2id.get(r['label'], -1))
        
        return {
            'boxes_3d': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 9), dtype=np.float32),
            'scores_3d': np.array(scores, dtype=np.float32) if scores else np.zeros((0,), dtype=np.float32),
            'labels_3d': np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        }

if __name__ == "__main__":
    with open(PKL_INFOS, 'rb') as f:
        data = pickle.load(f)

    # CAMBIO PROPUESTO:     
    #infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))

    runner = PillarnestHailoRunner(MODEL_PATH, WEIGHTS_NUMPY)
    final_results = runner.run(infos, limit=LIMIT_FRAMES)
    
    with open("hailo_results_sync.pkl", 'wb') as f:
        pickle.dump(final_results, f)
    print(f"✅ Éxito. Resultados SÍNCRONOS guardados en hailo_results_sync.pkl listos para evaluar.")
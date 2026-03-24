import os
import sys
import pickle
import copy
import numpy as np
from tqdm import tqdm
# Importamos esto por si quieres habilitar algún dibujo puntual, pero lo dejaremos apagado
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from hailo_platform import VDevice, HEF, FormatType, HailoSchedulingAlgorithm

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
MODEL_PATH = "model/hailo8l/pillarnest_tiny_def/pillarnest_tiny_original_export.hef"

# PONER A None PARA EL DATASET COMPLETO
LIMIT_FRAMES = None 
SAVE_VISUALS = False # Apagado para ir a máxima velocidad

output_path = "debug_hw_official"
output_path_images = os.path.join(output_path, "images")
os.makedirs(output_path_images, exist_ok=True)

class PillarnestHailoEvaluator:
    def __init__(self, model_path, weights_path):
        print(f"✅ Inicializando Evaluador...")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice_params = params
        self.model_path = model_path
        
        self.loader = PillarnestLoader(load_dim=5)
        self.sweeper = PillarnestMultiSweep(sweeps_num=Cfg.sweeps_num, remove_close=True, test_mode=True)
        self.voxelizer = PillarnestVoxelizer(Cfg.voxel_size, Cfg.point_cloud_range, Cfg.max_num_points, Cfg.max_voxels[1])
        self.encoder = PillarnestHeightEncoder(Cfg.voxel_size, Cfg.point_cloud_range, [Cfg.feat_channels])
        self.encoder.set_weights(np.load(weights_path, allow_pickle=True).item())
        self.scatter = PillarnestScatter(output_shape=Cfg.grid_size[:2], num_input_features=48)
        self.post_processor = CenterPointPostProcessor(Cfg)

    def run(self, val_infos, limit=None):
        results_list = []
        val_infos = val_infos[:limit]
        
        with VDevice(self.vdevice_params) as vdevice:
            self.hef = HEF(self.model_path)
            infer_model = vdevice.create_infer_model(self.model_path)
            infer_model.input().set_format_type(FormatType.FLOAT32)
            for output in infer_model.outputs:
                output.set_format_type(FormatType.FLOAT32)

            with infer_model.configure() as configured_model:
                bindings = configured_model.create_bindings()
                output_buffers = {name: np.empty(infer_model.output(name).shape, dtype=np.float32) 
                                 for name in infer_model.output_names}
                for name, buffer in output_buffers.items():
                    bindings.output(name).set_buffer(buffer)

                for idx, info in enumerate(tqdm(val_infos, desc="Full Inference")):
                    # 1. PRE-PROCESADO
                    frame_info = copy.deepcopy(info)
                    lidar_f = self._fix_path(frame_info['lidar_path'])
                    pts = self.loader.load_points(lidar_f)
                    for s in frame_info.get('sweeps', []): s['data_path'] = self._fix_path(s['data_path'])

                    pts_m = self.sweeper.process({'points': pts, 'timestamp': frame_info['timestamp']/1e6, 'sweeps': frame_info['sweeps']})['points']
                    v, c, n = self.voxelizer.voxelize(pts_m)
                    feat = self.encoder.encode(v, n, c)
                    canvas = self.scatter.scatter(feat, c)

                    # 2. INFERENCIA
                    bindings.input().set_buffer(np.ascontiguousarray(canvas, dtype=np.float32))
                    configured_model.run([bindings], 10000)

                    # 3. RECOGIDA
                    logical_output_map = {}
                    for vstream_name, buffer in output_buffers.items():
                        temp_4d = buffer if buffer.ndim == 4 else buffer[None, ...]
                        nchw_data = np.ascontiguousarray(temp_4d.transpose(0, 3, 1, 2))
                        orig_names = self.hef.get_original_names_from_vstream_name(vstream_name)
                        for name in orig_names: logical_output_map[name] = nchw_data

                    # 4. POST-PROCESADO
                    headers = ['reg', 'height', 'dim', 'rot', 'vel', 'iou', 'heatmap']
                    raw_res = []
                    for task_id in range(len(self.post_processor.tasks)):
                        for h in headers:
                            target_key = f"task{task_id}_{h}"
                            match = [v for k, v in logical_output_map.items() if target_key in k]
                            if match: raw_res.append(match[0])
                            else:
                                c = 2 if h in ['reg', 'rot', 'vel'] else 3 if h == 'dim' else 1
                                raw_res.append(np.zeros((1, c, 180, 180), dtype=np.float32))

                    preds = self.post_processor.forward(raw_res)

                    # 5. FORMATEAR PARA EVALUACIÓN
                    results_list.append({
                        'token': info['token'],
                        'boxes_3d': self._format_boxes(preds),
                        'scores_3d': np.array([p['score'] for p in preds]),
                        'labels_3d': self._format_labels(preds)
                    })

                    # Visualización opcional (solo si SAVE_VISUALS es True)
                    if SAVE_VISUALS and idx % 50 == 0:
                        self._draw_and_save(idx, pts_m, preds)

        return results_list

    def _format_boxes(self, preds):
        if not preds: return np.zeros((0, 9))
        boxes = []
        for p in preds:
            b = p['box']
            v = p['velocity']
            # Formato MMDet3D: [x, y, z, l, w, h, yaw, vx, vy]
            boxes.append([b['x'], b['y'], b['z'], b['l'], b['w'], b['h'], b['rot'], v[0], v[1]])
        return np.array(boxes)

    def _format_labels(self, preds):
        c2id = {name: i for i, name in enumerate(Cfg.class_names)}
        return np.array([c2id.get(p['label'], -1) for p in preds])

    def _fix_path(self, path):
        p = path.replace('\\', '/')
        f = 'samples' if 'samples' in p else 'sweeps' if 'sweeps' in p else ''
        return os.path.join(DATA_ROOT, f, p.split(f + '/')[-1]) if f else os.path.join(DATA_ROOT, os.path.basename(p))
    
    def _draw_and_save(self, idx, points, boxes):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        ax.scatter(points[::3, 0], points[::3, 1], s=0.1, c='gray', alpha=0.5)
        for b in boxes:
            if b['score'] < 0.25: continue
            c, s = np.cos(-b['box']['rot']), np.sin(-b['box']['rot'])
            R = np.array([[c, -s], [s, c]])
            corners = np.array([[b['box']['l']/2, b['box']['w']/2], [b['box']['l']/2, -b['box']['w']/2], 
                                [-b['box']['l']/2, -b['box']['w']/2], [-b['box']['l']/2, b['box']['w']/2]])
            corners = np.dot(corners, R.T) + np.array([b['box']['x'], b['box']['y']])
            ax.add_patch(patches.Polygon(corners, closed=True, edgecolor='orange', facecolor='none', linewidth=1.2))
        ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.axis('off')
        plt.savefig(f"{output_path_images}/frame_{idx:03d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    with open(PKL_INFOS, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    # CAMBIO PROPUESTO:     
    #infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    
    runner = PillarnestHailoEvaluator(MODEL_PATH, WEIGHTS_NUMPY)
    results = runner.run(infos, limit=LIMIT_FRAMES)
    
    # GUARDAR RESULTADOS
    if LIMIT_FRAMES is not None:
        print(f"\n⚠️  Solo se procesaron {LIMIT_FRAMES} frames. Guardando resultados parciales...")
        output_filename = f"hailo_8l_partial_results_{LIMIT_FRAMES}_frames.pkl"
    else:
        print(f"\n✅ Se procesaron todos los frames. Guardando resultados completos...")
        output_filename = "hailo_8l_full_results.pkl"

    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n🚀 ¡Inferencia completada! Guardado en {output_filename}")
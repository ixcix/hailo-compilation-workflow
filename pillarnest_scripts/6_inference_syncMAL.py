import os
import sys
import pickle
import copy
import numpy as np
from tqdm import tqdm

# --- VISUALIZACIÓN ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 🛠️ IMPORTANTE: Importamos InferVStreams para la inferencia síncrona segura
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
MODEL_PATH = "model/hailo8l/pillarnest_tiny_def/pillarnest_tiny_original_export.hef"

LIMIT_FRAMES = 5 

# ==========================================
# 🛠️ FUNCIONES AUXILIARES Y DE DIBUJO
# ==========================================
def fix_path(path, data_root=DATA_ROOT):
    p = path.replace('\\', '/')
    if 'samples' in p:
        rel = p.split('samples/')[-1]
        return os.path.join(data_root, 'samples', rel)
    if 'sweeps' in p:
        rel = p.split('sweeps/')[-1]
        return os.path.join(data_root, 'sweeps', rel)
    return os.path.join(data_root, os.path.basename(p))

def get_corners(x, y, w, l, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    return np.dot(corners, R.T) + np.array([x, y])

def draw_scene(ax, points, boxes, color, title):
    ax.set_facecolor('black')
    ax.scatter(points[::3, 0], points[::3, 1], s=0.1, c='gray', alpha=0.5)
    
    drawn_count = 0
    for b in boxes:
        if b['score'] < 0.25: continue
        
        box = b['box']
        x, y, rot = box['x'], box['y'], box['rot']
        w, l = box['w'], box['l']
        
        rot = -rot # NuScenes Fix
        corners = get_corners(x, y, w, l, rot)
        poly = patches.Polygon(corners, closed=True, edgecolor=color, facecolor='none', linewidth=1.2)
        ax.add_patch(poly)
        
        head = np.array([x, y]) + np.array([np.cos(rot), np.sin(rot)]) * (l/2)
        ax.plot([x, head[0]], [y, head[1]], color=color, linewidth=1)
        drawn_count += 1

    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    # Límite del mapa para ver si las cajas se chocan con la frontera
    rect = patches.Rectangle((-54, -54), 108, 108, linewidth=1, edgecolor='red', facecolor='none', linestyle='dashed')
    ax.add_patch(rect)
    ax.set_title(f"{title}\n(Mostrando {drawn_count} cajas seguras)", color='white')
    ax.axis('off')


# ==========================================
# 🚀 CLASE PRINCIPAL DE INFERENCIA SÍNCRONA
# ==========================================
class PillarnestHailoRunner:
    def __init__(self, model_path, weights_path):
        print(f"✅ Inicializando HailoRT (Síncrono)...")
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
        
        # Crear directorio para debug visual
        os.makedirs("debug_hw_sync", exist_ok=True)

    def run(self, val_infos, limit=None):
        results_mmdet = []
        val_infos = val_infos[:limit] if limit else val_infos
        
        output_nodes_info = {}
        for info in self.hef.get_output_vstream_infos():
            output_nodes_info[info.name] = {
                "original_names": self.hef.get_original_names_from_vstream_name(info.name),
                "shape": info.shape
            }

        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        input_vstream_name = input_vstream_info.name
        
        print(f"ℹ️ INFO DEL HEF: El chip espera una entrada con shape: {input_vstream_info.shape}")

        v_in_p = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        v_out_p = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)

        with InferVStreams(self.network_group, v_in_p, v_out_p) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):

            
                for idx, info in enumerate(tqdm(val_infos, desc="Inferencia Hailo Síncrona")):
                    # --- 1. PRE-PROCESADO ---
                    lidar_f = fix_path(info['lidar_path'])
                    if not os.path.exists(lidar_f): 
                        continue
                    
                    pts = self.loader.load_points(lidar_f)
                    sweeps_fixed = []
                    for s in info.get('sweeps', []):
                        s_copy = copy.deepcopy(s)
                        s_copy['data_path'] = fix_path(s['data_path'])
                        sweeps_fixed.append(s_copy)


                    pts_m = self.sweeper.process({
                        'points': pts, 'timestamp': info['timestamp']/1e6, 
                        'sweeps': sweeps_fixed, 'lidar_path': lidar_f
                    })['points']
                    
                    v, c, n = self.voxelizer.voxelize(pts_m)
                    feat = self.encoder.encode(v, n, c)
                    canvas = self.scatter.scatter(feat, c)

                    # --- 🕵️‍♂️ DEBUG 1: EL INPUT QUE ENTRA AL CHIP ---
                    # Si esto es igual en cada frame, el problema es el PRE-PROCESADO (sweeps/voxelizer)
                    print(f"\n🔍 [FRAME {idx}] INPUT CHECK:")
                    print(f"   Sum pts_m: {np.sum(pts_m):.2f} | Num pts: {len(pts_m)}")
                    print(f"   Canvas Mean: {canvas.mean():.6f} | Canvas Max: {canvas.max():.4f}")

                    input_data = np.expand_dims(canvas, axis=0).astype(np.float32)
                    input_data = np.ascontiguousarray(input_data)
                    print(f"DEBUG MEMORIA - Frame {idx} - Puntero ID: {id(input_data)}")
                    # Ejecutar inferencia
                    infer_results = infer_pipeline.infer({input_vstream_name: input_data})
                    
                    # --- 🕵️‍♂️ DEBUG 2: LA SALIDA QUE ESCUPE EL CHIP ---
                    # Si esto es igual en cada frame, el chip está "congelado" o el buffer de HailoRT no se limpia
                    print(f"🚀 [FRAME {idx}] OUTPUT CHECK (Raw from Chip):")
                    for name, tensor in infer_results.items():
                        # Miramos el primer heatmap (task0) para no saturar la consola
                        if "task0_heatmap" in name or "0" in name: 
                            print(f"   Stream {name} -> Mean: {tensor.mean():.6f} | Std: {tensor.std():.6f} | Max: {tensor.max():.4f}")

                    # --- REORGANIZACIÓN DE MEMORIA ---
                    output_map = {}
                    # ... (tu código de transposición) ...

                    # --- 🕵️‍♂️ DEBUG 3: LO QUE LLEGA AL POST-PROCESADOR ---
                    print(f"📦 [FRAME {idx}] POST-PROC CHECK:")
                    test_key = "task0_heatmap"
                    if test_key in output_map:
                        print(f"   {test_key} NCHW -> Mean: {output_map[test_key].mean():.6f} | Max: {output_map[test_key].max():.4f}")
                
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
        
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    
    runner = PillarnestHailoRunner(MODEL_PATH, WEIGHTS_NUMPY)
    final_results = runner.run(infos, limit=LIMIT_FRAMES)
    
    with open("hailo_results_sync.pkl", 'wb') as f:
        pickle.dump(final_results, f)
    print(f"✅ Éxito. Resultados SÍNCRONOS guardados. Revisa la carpeta 'debug_hw_sync' para ver las imágenes.")
import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import copy

# Hailo SDK
from hailo_sdk_client import ClientRunner, InferenceContext

# ==========================================
# IMPORTACIONES ADAPTADAS A POINTPILLARS
# ==========================================
# Asegúrate de que los nombres de los archivos y clases coincidan con tu entorno local
from pointpillars_logic_pre import (
    PointPillarsLoader, PointPillarsMultiSweep, PointPillarsVoxelizer, 
    PointPillarsEncoder, PointPillarsScatter
)
from pointpillars_logic_post import PointPillarsPostProcessor
from pointpillars_config import PointPillarsConfig as Cfg

PP_OUTPUTS = ['cls_score', 'bbox_pred', 'dir_cls_pred']

# ==========================================
# UTILIDADES DE RUTA Y GEOMETRÍA
# ==========================================
def fix_path(path, data_root):
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

# ==========================================
# CLASE DE INFERENCIA DUAL (EMULADOR)
# ==========================================
class HailoEmulatorComparator:
    def __init__(self, har_path, q_har_path):
        print(f"🔄 Cargando corredores de Hailo...")
        self.runner_fp = ClientRunner(har=har_path)
        self.runner_int8 = ClientRunner(har=q_har_path)
        
        # Mapeo de nombres originales
        self.fp_name_map = self._get_output_mapping(self.runner_fp)
        self.int8_name_map = self._get_output_mapping(self.runner_int8)

        try: 
            self.runner_fp._sdk_backend.update_fp_model(self.runner_fp._sdk_backend.model)
        except: 
            pass

    def _get_output_mapping(self, runner):
        hn = runner.get_hn_model()
        mapping = {}
        for layer in hn.get_output_layers():
            if layer.original_names:
                mapping[layer.name] = layer.original_names[0]
        return mapping

    def run_inference(self, input_tensor, runner, context_type):
        name_map = self.fp_name_map if "optimized" in str(context_type).lower() else self.int8_name_map

        with runner.infer_context(context_type) as ctx:
            model = runner.get_keras_model(ctx)
            raw_outputs = model(input_tensor)
            
            if not isinstance(raw_outputs, list):
                raw_outputs = [raw_outputs]
            
            keras_output_names = getattr(model, 'output_names', None)
            
            if keras_output_names is None:
                keras_output_names = list(name_map.keys())

            processed_outputs = {}
            for i, data in enumerate(raw_outputs):
                name_hailo = keras_output_names[i]
                original_name = name_map.get(name_hailo, name_hailo)
                
                val = data.numpy() if hasattr(data, 'numpy') else np.array(data)
                
                # Transponer de NHWC a NCHW
                if val.ndim == 4:
                    val = val.transpose(0, 3, 1, 2)
                
                # Intentamos forzar el nombre a uno de los conocidos de PP si hay coincidencia
                for target_key in PP_OUTPUTS:
                    if target_key in original_name:
                        original_name = target_key
                        break
                        
                processed_outputs[original_name] = val
            
            return processed_outputs

# ==========================================
# FUNCIÓN DE DIBUJO
# ==========================================
def draw_scene(ax, points, boxes, color, title):
    ax.set_facecolor('black')
    ax.scatter(points[::3, 0], points[::3, 1], s=0.1, c='gray', alpha=0.5)
    
    for b in boxes:
        if b['score'] < 0.25: 
            continue
        
        box = b['box']
        x, y, rot = box['x'], box['y'], box['rot']
        w, l = box['w'], box['l']
        
        rot = -rot
        corners = get_corners(x, y, w, l, rot)
        poly = patches.Polygon(corners, closed=True, edgecolor=color, facecolor='none', linewidth=1.2)
        ax.add_patch(poly)
        
        head = np.array([x, y]) + np.array([np.cos(rot), np.sin(rot)]) * (l/2)
        ax.plot([x, head[0]], [y, head[1]], color=color, linewidth=1)

    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.set_title(title, color='white')
    ax.axis('off')

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--har', required=True, help="Modelo FP32 (.har)")
    parser.add_argument('--qhar', required=True, help="Modelo INT8 (.har o .q.har)")
    parser.add_argument('--weights', required=False, default="model/hailo8l/pointpillars/pointpillars_encoder_weights.npy", help="Pesos del encoder (.npy)")
    parser.add_argument('--infos', required=False, default="data/nuscenes/v1.0-trainval/nuscenes_infos_val.pkl", help="val_infos.pkl")
    parser.add_argument('--data', required=False, default="data/nuscenes/v1.0-trainval", help="Carpeta raíz de NuScenes")
    parser.add_argument('--idx', type=int, default=0, help="Índice del frame a visualizar")
    args = parser.parse_args()

    # 1. Inicializar componentes de Pre/Post
    loader = PointPillarsLoader(load_dim=5)
    sweeper = PointPillarsMultiSweep(sweeps_num=Cfg.sweeps_num, remove_close=True, test_mode=True)
    voxelizer = PointPillarsVoxelizer(Cfg.voxel_size, Cfg.point_cloud_range, Cfg.max_num_points, Cfg.max_voxels[1])
    encoder = PointPillarsEncoder(Cfg.voxel_size, Cfg.point_cloud_range, [Cfg.feat_channels])
    
    if os.path.exists(args.weights):
        encoder.set_weights(np.load(args.weights, allow_pickle=True).item())
        
    scatter = PointPillarsScatter(output_shape=Cfg.grid_size[:2], num_input_features=64)
    post_processor = PointPillarsPostProcessor(Cfg)

    # 2. Cargar Frame específico
    with open(args.infos, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    info = infos[args.idx]
    
    print(f"📦 Procesando Frame {args.idx} (Token: {info.get('token', 'N/A')})")
    
    lidar_path = fix_path(info['lidar_path'], args.data)
    pts = loader.load_points(lidar_path)
    
    info_copy = copy.deepcopy(info)
    if 'sweeps' in info_copy:
        for s in info_copy['sweeps']:
            s['data_path'] = fix_path(s['data_path'], args.data)
        
        pts_m = sweeper.process({
            'points': pts,
            'timestamp': info_copy['timestamp'] / 1e6,
            'sweeps': info_copy['sweeps'],
            'lidar_path': lidar_path
        })['points']
    else:
        pts_m = pts
    
    print("📍 Lidar path:", lidar_path)

    v, c, n = voxelizer.voxelize(pts_m)
    feat = encoder.encode(v, n, c)
    canvas = scatter.scatter(feat, c) 
    
    input_tensor = np.expand_dims(canvas, axis=0).astype(np.float32)
    print("Input tensor:", input_tensor.shape)
    
    # 3. Inferencias en Emulador
    comparator = HailoEmulatorComparator(args.har, args.qhar)
    
    print("🔵 Ejecutando FP32...")
    out_fp = comparator.run_inference(input_tensor, comparator.runner_fp, InferenceContext.SDK_FP_OPTIMIZED)
    
    print("🔴 Ejecutando INT8...")
    out_int8 = comparator.run_inference(input_tensor, comparator.runner_int8, InferenceContext.SDK_QUANTIZED)

    # 4. Post-procesado unificado para PointPillars
    def decode(output_dict):
        # Nos aseguramos de extraer los tensores correctos usando la lista oficial de PP
        flat_list = []
        for key in PP_OUTPUTS:
            if key in output_dict:
                flat_list.append(output_dict[key])
            else:
                flat_list.append(np.zeros((1, 1, 1, 1)))
        
        # Dependiendo de tu implementación de PointPillarsPostProcessor, 
        # puede que requiera los argumentos por nombre o como lista plana.
        return post_processor.forward(flat_list)

    boxes_fp = decode(out_fp)
    boxes_int8 = decode(out_int8)

    # 5. Visualización Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='black')
    
    draw_scene(axes[0], pts_m, boxes_fp, 'cyan', f"FP32 Emulator (Detecciones: {len(boxes_fp)})")
    draw_scene(axes[1], pts_m, boxes_int8, 'orange', f"INT8 Emulator (Detecciones: {len(boxes_int8)})")

    plt.tight_layout()
    model_name = os.path.basename(args.qhar).split('.')[0]
    out_img = f"debug/comp_frame_{model_name}_{args.idx}.png"
    os.makedirs("debug", exist_ok=True)
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    print(f"✅ Comparativa guardada en: {out_img}")

if __name__ == "__main__":
    main()
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

# --- Tus módulos de lógica ---
# Asegúrate de que las rutas sean correctas según tu estructura de carpetas
from pillarnest_logic_pre import (
    PillarnestLoader, PillarnestMultiSweep, PillarnestVoxelizer, 
    PillarnestHeightEncoder, PillarnestScatter
)
from pillarnest_logic_post import CenterPointPostProcessor
from pillarnest_config import PillarnestTinyConfig as Cfg

# ==========================================
# 🛠️ UTILIDADES DE RUTA Y GEOMETRÍA
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
# 🧠 CLASE DE INFERENCIA DUAL (EMULADOR)
# ==========================================
class HailoEmulatorComparator:
    def __init__(self, har_path, q_har_path):
        print(f"🔄 Cargando corredores de Hailo...")
        self.runner_fp = ClientRunner(har=har_path)
        self.runner_int8 = ClientRunner(har=q_har_path)
        
        # Mapeo de nombres: de "Nombre Keras/Interno" a "Original Name"
        self.fp_name_map = self._get_output_mapping(self.runner_fp)
        self.int8_name_map = self._get_output_mapping(self.runner_int8)

        try: self.runner_fp._sdk_backend.update_fp_model(self.runner_fp._sdk_backend.model)
        except: pass

    def _get_output_mapping(self, runner):
        """ Crea un mapa: {nombre_capa_hailo: nombre_original_tuyo} """
        hn = runner.get_hn_model()
        mapping = {}
        for layer in hn.get_output_layers():
            # Mapeamos el nombre de la capa al primer nombre original (ej: 'task0_reg')
            if layer.original_names:
                mapping[layer.name] = layer.original_names[0]
        return mapping

    def run_inference(self, input_tensor, runner, context_type):
        """
        Ejecuta el emulador y devuelve un dict en formato NCHW con nombres originales.
        """
        # Elegir el mapa de nombres adecuado
        name_map = self.fp_name_map if "optimized" in str(context_type).lower() else self.int8_name_map

        with runner.infer_context(context_type) as ctx:
            model = runner.get_keras_model(ctx)
            raw_outputs = model(input_tensor)
            
            # Asegurar que raw_outputs sea una lista
            if not isinstance(raw_outputs, list):
                raw_outputs = [raw_outputs]
            
            # --- SOLUCIÓN AL NONETYPE ---
            # Si model.output_names falla, usamos las llaves de nuestro name_map
            # que extrajimos directamente del HN (grafo de Hailo)
            keras_output_names = getattr(model, 'output_names', None)
            
            if keras_output_names is None:
                # Fallback: Usamos los nombres de las capas del HN en orden
                keras_output_names = list(name_map.keys())

            processed_outputs = {}
            for i, data in enumerate(raw_outputs):
                # Intentamos sacar el nombre por índice si los nombres de Keras fallan
                name_hailo = keras_output_names[i]
                
                # Buscamos el nombre original (taskX_...) en nuestro mapa
                original_name = name_map.get(name_hailo, name_hailo)
                
                # Convertir a Numpy
                val = data.numpy() if hasattr(data, 'numpy') else np.array(data)
                
                # Transponer de NHWC a NCHW
                if val.ndim == 4:
                    val = val.transpose(0, 3, 1, 2)
                
                processed_outputs[original_name] = val
            
            return processed_outputs

# ==========================================
# 🎨 FUNCIÓN DE DIBUJO
# ==========================================
def draw_scene(ax, points, boxes, color, title):
    ax.set_facecolor('black')
    # Dibujamos solo una parte de los puntos para no saturar
    ax.scatter(points[::3, 0], points[::3, 1], s=0.1, c='gray', alpha=0.5)
    
    for b in boxes:
        if b['score'] < 0.25: continue
        
        box = b['box']
        x, y, rot = box['x'], box['y'], box['rot']
        w, l = box['w'], box['l']
        
        # NuScenes Fix: Intercambio L/W para visualización si es necesario
        # l, w = w, l 
        rot = -rot
        #prueba rotacion
        # x, y = y, x
        corners = get_corners(x, y, w, l, rot)
        poly = patches.Polygon(corners, closed=True, edgecolor=color, facecolor='none', linewidth=1.2)
        ax.add_patch(poly)
        
        # Dirección
        head = np.array([x, y]) + np.array([np.cos(rot), np.sin(rot)]) * (l/2)
        ax.plot([x, head[0]], [y, head[1]], color=color, linewidth=1)

    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.set_title(title, color='white')
    ax.axis('off')

# ==========================================
# 🚀 SCRIPT PRINCIPAL
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--har', required=True, help="Modelo FP32 (.har)")
    parser.add_argument('--qhar', required=True, help="Modelo INT8 (.har o .q.har)")
    parser.add_argument('--weights', required=False, default="/local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_original/pillarnest_encoder_weights.npy", help="Pesos del encoder (.npy)")
    parser.add_argument('--infos', required=False, default="/local/shared_with_docker/hailo-compilation-workflow/data/nuscenes/v1.0-trainval/nuscenes_infos_val.pkl", help="val_infos.pkl")
    parser.add_argument('--data', required=False, default="/local/shared_with_docker/hailo-compilation-workflow/data/nuscenes/v1.0-trainval", help="Carpeta raíz de NuScenes")
    parser.add_argument('--idx', type=int, default=0, help="Índice del frame a visualizar")
    args = parser.parse_args()

    # 1. Inicializar componentes de Pre/Post
    loader = PillarnestLoader(load_dim=5)
    sweeper = PillarnestMultiSweep(sweeps_num=Cfg.sweeps_num, remove_close=True, test_mode=True)
    voxelizer = PillarnestVoxelizer(Cfg.voxel_size, Cfg.point_cloud_range, Cfg.max_num_points, Cfg.max_voxels[1])
    encoder = PillarnestHeightEncoder(Cfg.voxel_size, Cfg.point_cloud_range, [Cfg.feat_channels])
    encoder.set_weights(np.load(args.weights, allow_pickle=True).item())
    scatter = PillarnestScatter(output_shape=Cfg.grid_size[:2], num_input_features=48)
    post_processor = CenterPointPostProcessor(Cfg)

    # 2. Cargar Frame específico
    with open(args.infos, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    info = infos[args.idx]
    
    print(f"📦 Procesando Frame {args.idx} - Token: {info.get('token', 'N/A')}")
    
    # Pre-procesado real (Multisweep)
    lidar_path = fix_path(info['lidar_path'], args.data)
    pts = loader.load_points(lidar_path)
    
    # Importante: Usamos la info del pkl para los sweeps
    info = copy.deepcopy(infos[args.idx])
    for s in info['sweeps']:
        s['data_path'] = fix_path(s['data_path'], args.data)
    
    pts_m = sweeper.process({
        'points': pts,
        'timestamp': info['timestamp'] / 1e6,
        'sweeps': info['sweeps'],
        'lidar_path': lidar_path
    })['points']
    
    print("📍 Lidar path:", lidar_path)
    print("📍 Sweeps:", len(info['sweeps']))
    for i, s in enumerate(info['sweeps'][:2]):
        print(f"   sweep {i}: {s['data_path']}")


    v, c, n = voxelizer.voxelize(pts_m)
    feat = encoder.encode(v, n, c)
    canvas = scatter.scatter(feat, c) # [720, 720, 48] (NHWC)
    
    input_tensor = np.expand_dims(canvas, axis=0).astype(np.float32)
    print("Input tensor:", input_tensor.shape)
    print("Voxelized points:", pts_m.shape)
    # 3. Inferencias en Emulador
    comparator = HailoEmulatorComparator(args.har, args.qhar)
    
    print("🔵 Ejecutando FP32...")
    out_fp = comparator.run_inference(input_tensor, comparator.runner_fp, InferenceContext.SDK_FP_OPTIMIZED)
    
    print("🔴 Ejecutando INT8...")
    out_int8 = comparator.run_inference(input_tensor, comparator.runner_int8, InferenceContext.SDK_QUANTIZED)

    # 4. Post-procesado unificado
    def decode(output_dict):
        flat_list = []
        for task_id in range(len(post_processor.tasks)):
            for h in post_processor.headers:
                key = f"task{task_id}_{h}"
                # Buscamos la clave ignorando prefijos de nodos si los hay
                match = [v for k, v in output_dict.items() if key in k]
                flat_list.append(match[0] if match else np.zeros((1, 1, 1, 1)))
        return post_processor.forward(flat_list)

    boxes_fp = decode(out_fp)
    boxes_int8 = decode(out_int8)

    # 5. Visualización Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='black')
    
    draw_scene(axes[0], pts_m, boxes_fp, 'cyan', f"FP32 Emulator (Detecciones: {len(boxes_fp)})")
    draw_scene(axes[1], pts_m, boxes_int8, 'orange', f"INT8 Emulator (Detecciones: {len(boxes_int8)})")

    plt.tight_layout()
    model = os.path.basename(args.qhar).split('.')[0]
    out_img = f"debug/comp_frame_{model}_{args.idx}.png"
    os.makedirs("debug", exist_ok=True)
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    print(f"✅ Comparativa guardada en: {out_img}")
    plt.show()

if __name__ == "__main__":
    main()
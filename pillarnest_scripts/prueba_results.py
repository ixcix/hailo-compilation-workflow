import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

sys.path.append("/local/shared_with_docker/hailo-compilation-workflow")
from model.hailo8l.pillarnest_retraining.pillarnest_logic_pre import PillarnestLoader, PillarnestMultiSweep
from model.hailo8l.pillarnest_retraining.pillarnest_config import PillarnestTinyConfig as Cfg

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
DATA_ROOT = "/local/shared_with_docker/hailo-compilation-workflow/data/nuscenes/v1.0-trainval"
INFOS_PKL = f"{DATA_ROOT}/nuscenes_infos_val.pkl"
RESULTS_PKL = "hailo_results_sync.pkl"

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

def draw_scene(ax, points, boxes_data, color, title):
    ax.set_facecolor('black')
    # Dibujamos nube de puntos (diezmada para no saturar)
    ax.scatter(points[::3, 0], points[::3, 1], s=0.1, c='gray', alpha=0.5)
    
    boxes = boxes_data['boxes_3d']
    scores = boxes_data['scores_3d']
    
    drawn_count = 0
    for i in range(len(boxes)):
        if scores[i] < 0.01:  # Filtro visual para ver solo lo más seguro
            continue
            
        # Formato de boxes_3d NumPy: [x, y, z, l, w, h, rot, vx, vy]
        box = boxes[i]
        x, y, rot = box[0], box[1], box[6]
        l, w = box[3], box[4]
        
        # NuScenes Fix de tu script original
        rot = -rot
        
        corners = get_corners(x, y, w, l, rot)
        poly = patches.Polygon(corners, closed=True, edgecolor=color, facecolor='none', linewidth=1.2)
        ax.add_patch(poly)
        
        # Dirección (Heading)
        head = np.array([x, y]) + np.array([np.cos(rot), np.sin(rot)]) * (l/2)
        ax.plot([x, head[0]], [y, head[1]], color=color, linewidth=1)
        drawn_count += 1

    # Límites del rango del Point Cloud configurado (-54 a +54)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    
    # Dibujar líneas rojas para marcar el borde del rango de percepción
    rect = patches.Rectangle((-54, -54), 108, 108, linewidth=1, edgecolor='red', facecolor='none', linestyle='dashed')
    ax.add_patch(rect)

    ax.set_title(f"{title}\n(Mostrando {drawn_count} cajas con Score > 0.25)", color='white')
    ax.axis('off')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0, help="Índice del frame a visualizar")
    args = parser.parse_args()

    # 1. Cargar Infos Oficiales y Nube de Puntos
    loader = PillarnestLoader(load_dim=5)
    sweeper = PillarnestMultiSweep(sweeps_num=Cfg.sweeps_num, remove_close=True, test_mode=True)
    
    with open(INFOS_PKL, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    info = infos[args.idx]
    
    print(f"📦 Procesando Frame {args.idx} - Token: {info.get('token', 'N/A')}")
    
    lidar_path = fix_path(info['lidar_path'], DATA_ROOT)
    pts = loader.load_points(lidar_path)
    
    info_copy = copy.deepcopy(info)
    for s in info_copy.get('sweeps', []):
        s['data_path'] = fix_path(s['data_path'], DATA_ROOT)
    
    pts_m = sweeper.process({
        'points': pts,
        'timestamp': info_copy['timestamp'] / 1e6,
        'sweeps': info_copy.get('sweeps', []),
        'lidar_path': lidar_path
    })['points']
    
    # 2. Cargar tus resultados de inferencia (.pkl)
    with open(RESULTS_PKL, 'rb') as f:
        hailo_results = pickle.load(f)
        
    # Verificar si el PKL tiene el mismo tamaño
    if len(hailo_results) <= args.idx:
        print(f"❌ Error: El índice {args.idx} no existe en los resultados. (Max: {len(hailo_results)-1})")
        return

    frame_result = hailo_results[args.idx]['pts_bbox']

    # 3. Dibujar
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='black')
    
    draw_scene(ax, pts_m, frame_result, 'orange', f"Resultados Hailo PKL (Frame {args.idx})")

    plt.tight_layout()
    out_img = f"debug_pkl_frame_{args.idx}.png"
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    print(f"✅ Imagen guardada en: {out_img}")

if __name__ == "__main__":
    main()
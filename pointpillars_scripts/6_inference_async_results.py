import os
import sys
import pickle
import copy
import numpy as np
from tqdm import tqdm
import time
import psutil
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from hailo_platform import VDevice, HEF, FormatType, HailoSchedulingAlgorithm

# Asegúrate de que esta ruta apunte a donde tienes tus scripts de PointPillars
sys.path.append("/local/shared_with_docker/hailo-compilation-workflow")
from pointpillars_scripts.pointpillars_logic_pre import (
    Loader, MultiSweep, Voxelizer, 
    PointpillarsHardVFE, Scatter
)
from pointpillars_scripts.pointpillars_logic_post import PointPillarsPostProcessor
import pointpillars_scripts.pointpillars_config as cfg

# ==========================================
# CONFIGURACIÓN
# ==========================================
PKL_INFOS = "/local/shared_with_docker/hailo-compilation-workflow/data/nuscenes/v1.0-trainval/nuscenes_infos_val.pkl"
DATA_ROOT = "/local/shared_with_docker/hailo-compilation-workflow/data/nuscenes/v1.0-trainval"
WEIGHTS_NUMPY = "model/hailo8l/pointpillars/pointpillars_vfe_weights.npy" 
MODEL_PATH = "model/hailo8l/pointpillars/pointpillars_opt0_base.hef"

# PONER A None PARA EL DATASET COMPLETO
LIMIT_FRAMES = 5
SAVE_VISUALS = True 

output_path = "inference_output"
output_path_images = os.path.join(output_path, "images")
os.makedirs(output_path_images, exist_ok=True)

# Usamos la configuración de PointPillars
Cfg = cfg.PointPillarsConfig

class PointPillarsHailoEvaluator:
    def __init__(self, model_path, weights_path):
        print(f"✅ Inicializando Evaluador PointPillars (Hailo-8L)...")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice_params = params
        self.model_path = model_path
        
        # Módulos Pre-procesado PointPillars
        self.loader = Loader(load_dim=5)
        self.sweeper = MultiSweep(sweeps_num=Cfg.sweeps_num, remove_close=True, test_mode=True)
        self.voxelizer = Voxelizer(Cfg.voxel_size, Cfg.point_cloud_range, Cfg.max_num_points, Cfg.max_voxels)
        
        # Encoder específico de PointPillars
        self.encoder = PointpillarsHardVFE(Cfg.voxel_size, Cfg.point_cloud_range)
        self.encoder.set_weights(np.load(weights_path, allow_pickle=True).item())
        
        self.scatter = Scatter(output_shape=Cfg.output_shape, num_input_features=Cfg.in_channels_scatter)
        
        # Post-procesador que acabamos de validar
        self.post_processor = PointPillarsPostProcessor(Cfg)

        self.stats = {
            't_pre': [], 't_infer': [], 't_post': [], 't_total': [],
            'cpu_util': [], 'npu_util': [], 'hailo_temp': []
        }

    def run(self, val_infos, limit=None, csv_filename="hw_metrics_pointpillars.csv"):
        results_list = []
        if limit is not None:
            val_infos = val_infos[:limit]
        
        psutil.cpu_percent()

        with VDevice(self.vdevice_params) as vdevice:
            self.hef = HEF(self.model_path)
            infer_model = vdevice.create_infer_model(self.model_path)
            infer_model.input().set_format_type(FormatType.FLOAT32)
            for output in infer_model.outputs:
                output.set_format_type(FormatType.FLOAT32)

            try:
                phys_devs = vdevice.get_physical_devices()
                hailo_control = phys_devs[0].control if len(phys_devs) > 0 else None
            except Exception:
                hailo_control = None

            with infer_model.configure() as configured_model:
                bindings = configured_model.create_bindings()
                output_buffers = {name: np.empty(infer_model.output(name).shape, dtype=np.float32) 
                                 for name in infer_model.output_names}
                for name, buffer in output_buffers.items():
                    bindings.output(name).set_buffer(buffer)

                for idx, info in enumerate(tqdm(val_infos, desc="PointPillars Inference")):
                    t_frame_start = time.perf_counter()

                    # 1. PRE-PROCESADO (Modular PointPillars)
                    t_pre_start = time.perf_counter()
                    frame_info = copy.deepcopy(info)
                    lidar_f = self._fix_path(frame_info['lidar_path'])
                    pts = self.loader.load_points(lidar_f)
                    for s in frame_info.get('sweeps', []): s['data_path'] = self._fix_path(s['data_path'])

                    pts_m = self.sweeper.process({'points': pts, 'timestamp': frame_info['timestamp']/1e6, 'sweeps': frame_info['sweeps']})['points']
                    v, c, n = self.voxelizer.voxelize(pts_m)
                    feat = self.encoder.encode(v, n, c)
                    canvas = self.scatter.scatter(feat, c)
                    self.stats['t_pre'].append(time.perf_counter() - t_pre_start)

                    # 2. INFERENCIA NPU
                    t_infer_start = time.perf_counter()
                    # Canvas NHWC directo a la NPU
                    bindings.input().set_buffer(np.ascontiguousarray(canvas, dtype=np.float32))
                    configured_model.run([bindings], 10000)
                    self.stats['t_infer'].append(time.perf_counter() - t_infer_start)

                    # 3. POST-PROCESADO (Mapeo de 3 Salidas)
                    t_post_start = time.perf_counter()
                    logical_output_map = {}
                    for vstream_name, buffer in output_buffers.items():
                        temp_4d = buffer if buffer.ndim == 4 else buffer[None, ...]
                        # Hailo entrega NHWC, pasamos a NCHW para el post-processor
                        nchw_data = np.ascontiguousarray(temp_4d.transpose(0, 3, 1, 2))
                        orig_names = self.hef.get_original_names_from_vstream_name(vstream_name)
                        for name in orig_names: logical_output_map[name] = nchw_data

                    # Recolectamos las 3 cabezas específicas de PointPillars
                    # El orden debe ser: cls, bbox, dir
                    head_keys = ['cls_score', 'bbox_pred', 'dir_cls_pred']
                    raw_res = []
                    for h_key in head_keys:
                        # Buscamos el buffer que contenga el nombre de la cabeza
                        match = [v for k, v in logical_output_map.items() if h_key in k]
                        if match:
                            raw_res.append(match[0])
                        else:
                            print(f"⚠️ Error: No se encontró la salida {h_key} en el HEF")
                            # Fallback: tensor vacío con dimensiones NuScenes (200x200 o 400x400)
                            raw_res.append(np.zeros((1, 1, canvas.shape[0]//2, canvas.shape[1]//2)))

                    preds = self.post_processor.forward(raw_res)
                    self.stats['t_post'].append(time.perf_counter() - t_post_start)

                    # 4. TELEMETRÍA
                    self.stats['t_total'].append(time.perf_counter() - t_frame_start)
                    self.stats['cpu_util'].append(psutil.cpu_percent())
                    
                    if hailo_control:
                        try:
                            temp_obj = hailo_control.get_chip_temperature()
                            self.stats['hailo_temp'].append(getattr(temp_obj, 'ts0_temperature', getattr(temp_obj, 'temperature', 0)))
                        except Exception: pass

                    # FORMATEAR RESULTADOS
                    results_list.append({
                        'token': info['token'],
                        'boxes_3d': self._format_boxes(preds),
                        'scores_3d': np.array([p['score'] for p in preds]),
                        'labels_3d': self._format_labels(preds)
                    })
                    if SAVE_VISUALS and idx % 1 == 0:
                        self._draw_and_save(idx, pts_m, preds)


        self._print_and_save_hardware_report(csv_filename)
        return results_list

    def _print_and_save_hardware_report(self, csv_filename):
        # Mantenemos tu lógica de reporte idéntica para comparar con Pillarnest
        print("\n" + "="*65)
        print("📊 REPORTE HARDWARE: POINTPILLARS EN HAILO-8L")
        print("="*65)
        t_pre = np.mean(self.stats['t_pre']) * 1000
        t_inf = np.mean(self.stats['t_infer']) * 1000
        t_pos = np.mean(self.stats['t_post']) * 1000
        t_tot = np.mean(self.stats['t_total']) * 1000
        
        print(f"⏱️  Latencia Pre:   {t_pre:.2f} ms")
        print(f"⏱️  Latencia NPU:   {t_inf:.2f} ms")
        print(f"⏱️  Latencia Post:  {t_pos:.2f} ms")
        print(f"⏱️  Latencia Total: {t_tot:.2f} ms")
        print(f"🚀 FPS E2E:        {1000.0/t_tot:.2f}")
        print(f"🌡️  Temp Media:     {np.mean(self.stats['hailo_temp']):.2f} °C")
        
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Pre_ms", "NPU_ms", "Post_ms", "Total_ms", "FPS", "CPU_%", "Temp_C"])
            writer.writerow([t_pre, t_inf, t_pos, t_tot, 1000/t_tot, np.mean(self.stats['cpu_util']), np.mean(self.stats['hailo_temp'])])

    def _format_boxes(self, preds):
        if not preds: return np.zeros((0, 9))
        boxes = []
        for p in preds:
            b = p['box']
            v = p['velocity']
            # Formato NuScenes: [x, y, z, w, l, h, rot, vx, vy]
            boxes.append([b['x'], b['y'], b['z'], b['w'], b['l'], b['h'], b['rot'], v[0], v[1]])
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
    infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    
    name = MODEL_PATH.split('/')[-1].replace('.hef', '')
    
    # Nombres de archivos dinámicos según LIMIT_FRAMES
    if LIMIT_FRAMES is not None:
        output_filename = f"inference_output/{name}_partial_results_{LIMIT_FRAMES}_frames.pkl"
        csv_filename = f"inference_output/{name}_partial_metrics_{LIMIT_FRAMES}_frames.csv"
        print(f"\n⚠️  Ejecución limitada a {LIMIT_FRAMES} frames.")
    else:
        output_filename = f"inference_output/{name}_full_results.pkl"
        csv_filename = f"inference_output/{name}_full_metrics.csv"
        print(f"\n✅ Ejecución de dataset completo.")


    runner = PointPillarsHailoEvaluator(MODEL_PATH, WEIGHTS_NUMPY)
    results = runner.run(infos, limit=LIMIT_FRAMES, csv_filename=csv_filename)
    
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"🚀 Inferencia PointPillars completada. Resultados en {output_filename}")
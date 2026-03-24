import argparse
import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from hailo_sdk_client import ClientRunner, InferenceContext

# Importar lógica
try:
    import pillarnest_logic_singlesweep as pillarnest_logic
except ImportError:
    try:
        import pillarnest_logic
    except:
        print("❌ Error: No se encuentra la lógica de post-procesado.")
        sys.exit(1)

# prueba sin velocidad 
# headers = ['reg','height','dim','rot','iou','heatmap']

headers = ['reg','height','dim','rot', 'vel', 'iou','heatmap']

# ==========================================
# 1. CLASE WRAPPER (Tu código robusto)
# ==========================================
class PillarNest_Hailo_Module(torch.nn.Module):
    def __init__(self, runner, context_type):
        super().__init__()
        self._runner = runner
        print(f"⚙️  Contexto: {context_type}")
        with runner.infer_context(context_type) as ctx:
            self._hailo_model = runner.get_keras_model(ctx)

    def forward(self, input_tensor):        
        input_nhwc = np.transpose(input_tensor.cpu().detach().numpy(), (0, 2, 3, 1))
        try: raw_outputs = self._hailo_model({'input_pillars': input_nhwc})
        except: raw_outputs = self._hailo_model(input_nhwc)

        # Recuperación de nombres
        if isinstance(raw_outputs, list):
            keys = None
            try: keys = list(self._runner.get_hn_model().get_end_node_names())
            except: pass
            if not keys: keys = getattr(self._hailo_model, 'output_names', None)
            if (not keys) and hasattr(self._hailo_model, 'outputs') and self._hailo_model.outputs:
                 keys = [t.name.split(':')[0].split('/')[0] if hasattr(t, 'name') else str(t) for t in self._hailo_model.outputs]
            if keys is None or len(keys) != len(raw_outputs):
                keys = [f"task{i}_{h}" for i in range(6) for h in headers]
                if len(keys) != len(raw_outputs): keys = [f"out_{i}" for i in range(len(raw_outputs))]
            raw_outputs = dict(zip(keys, raw_outputs))

        # Convertir a NCHW
        clean_outputs = {}
        for k, v in raw_outputs.items():
            if hasattr(v, 'numpy'): v = v.numpy()
            else: v = np.array(v)
            clean_k = k.split(':')[0].split('/')[0]
            if v.ndim == 4: v = v.transpose(0, 3, 1, 2)
            clean_outputs[clean_k] = v
        return clean_outputs

# ==========================================
# 2. FUNCIONES DE VISUALIZACIÓN
# ==========================================
def get_corners(x, y, w, l, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    rotated = np.dot(corners, R.T) + np.array([x, y])
    return rotated

def draw_boxes(ax, boxes, color, label, linestyle='-'):
    """ Dibuja las cajas en el eje proporcionado """
    for box in boxes:
        x, y, z, w, l, h, rot, score, cls_id = box
        if score < 0.25: continue
        
        # FIX VISUAL W/L (Comenta si ya lo arreglaste en pillarnest_logic)
        l, w = w, l 
        
        corners = get_corners(x, y, w, l, rot)
        poly = patches.Polygon(corners, closed=True, edgecolor=color, facecolor='none', linewidth=1.5, linestyle=linestyle)
        ax.add_patch(poly)
        
        # Dirección
        center = np.array([x, y])
        head = center + np.array([np.cos(rot), np.sin(rot)]) * (l/2)
        ax.plot([center[0], head[0]], [center[1], head[1]], color=color, linewidth=1)
    
    # Devolvemos handle para la leyenda local
    return patches.Patch(color=color, label=f"{label} ({len(boxes)})")

def decode_boxes(output_dict, threshold=0.25):
    """ Decodifica las salidas de las 6 tareas """
    all_boxes = []
    for i in range(6):
        task_prefix = f"task{i}_"
        t_dict = {k.replace(task_prefix, ""): v for k,v in output_dict.items() if k.startswith(task_prefix)}
        if 'heatmap' in t_dict:
            for k in t_dict:
                if t_dict[k].ndim == 3: t_dict[k] = t_dict[k][None, ...]
            try:
                boxes = pillarnest_logic.postprocess_outputs(t_dict, score_thresh=threshold)
                if len(boxes) > 0:
                    boxes[:, 8] = i * 10 + boxes[:, 8]
                    all_boxes.append(boxes)
            except: pass
    if all_boxes: return np.concatenate(all_boxes)
    return np.empty((0, 9))

# ==========================================
# 3. MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--har', required=True)
    parser.add_argument('--qhar', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--golden', required=True)
    parser.add_argument('--bin', required=True)
    args = parser.parse_args()

    pillarnest_logic.load_weights('pillarnest_scripts/weights_singlesweep_finetune')

    print("📂 Cargando datos...")
    input_data = np.load(args.input)
    input_tensor = torch.from_numpy(input_data)
    golden_data = np.load(args.golden)
    try: points = np.fromfile(args.bin, dtype=np.float32).reshape(-1, 5)
    except: points = np.fromfile(args.bin, dtype=np.float32).reshape(-1, 4)

    # --- INFERENCIAS ---
    print("\n🔵 Ejecutando FP32...")
    boxes_fp = []
    try:
        runner_fp = ClientRunner(har=args.har)
        try: runner_fp._sdk_backend.update_fp_model(runner_fp._sdk_backend.model)
        except: pass
        mod_fp = PillarNest_Hailo_Module(runner_fp, InferenceContext.SDK_FP_OPTIMIZED)
        boxes_fp = decode_boxes(mod_fp(input_tensor))
    except Exception as e: print(f"   ⚠️ Falló FP32: {e}")

    print("\n🔴 Ejecutando INT8...")
    boxes_q = []
    try:
        runner_q = ClientRunner(har=args.qhar)
        mod_q = PillarNest_Hailo_Module(runner_q, InferenceContext.SDK_QUANTIZED)
        boxes_q = decode_boxes(mod_q(input_tensor))
    except Exception as e: print(f"   ⚠️ Falló INT8: {e}")

    print("\n🟢 Procesando Golden...")
    boxes_golden = decode_boxes({k: v for k, v in golden_data.items()})

    # ==========================================
    # --- PLOT PARALELO (SIDE-BY-SIDE) ---
    # ==========================================
    print(f"\n📊 Generando gráfico paralelo...")
    # Creamos una figura ancha con 3 subplots en fila
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='black')
    
    # Definimos las configuraciones de cada panel
    scenarios = [
        {"ax": axes[0], "boxes": boxes_fp,     "color": 'cyan', "title": "FP32 (Emulador SDK)", "style": '-'},
        {"ax": axes[1], "boxes": boxes_q,      "color": 'red',  "title": "INT8 (Quantized)",     "style": '-'},
        {"ax": axes[2], "boxes": boxes_golden, "color": 'lime', "title": "Golden (PyTorch)",    "style": '--'}
    ]

    # Iteramos y dibujamos cada panel
    for config in scenarios:
        ax = config["ax"]
        boxes = config["boxes"]
        color = config["color"]
        title = config["title"]
        style = config["style"]

        ax.set_facecolor('black')
        # Nube de puntos de fondo (igual en los tres)
        ax.scatter(points[::5, 0], points[::5, 1], s=0.1, c='gray', alpha=0.5)
        
        handles = []
        if len(boxes) > 0:
            h = draw_boxes(ax, boxes, color, title.split()[0], style)
            handles.append(h)
            ax.legend(handles=handles, loc='upper right')
        else:
             ax.text(0, 0, "Sin Detecciones", color='white', ha='center')

        # Configuración común
        ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
        ax.set_title(title, color='white', fontsize=14)
        ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')

    # Título global y guardado
    frame_id = os.path.basename(args.bin)
    fig.suptitle(f"Comparativa Paralela de Detección | Frame: {frame_id}", color='white', fontsize=16, y=0.99)
    plt.tight_layout()
    
    out_file = "parallel_comparison.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"🖼️  Imagen paralela guardada en: {out_file}")

if __name__ == "__main__":
    main()
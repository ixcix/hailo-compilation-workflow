import argparse
import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from hailo_sdk_client import ClientRunner, InferenceContext

# --- Importamos tu NUEVA lógica unificada ---
try:
    from pillarnest_logic_post import CenterPointPostProcessor
    from pillarnest_config import PillarnestTinyConfig as Cfg
except ImportError:
    print("❌ Error: No se encuentran 'pillarnest_logic_post.py' o 'pillarnest_config.py'. Asegúrate de ejecutar desde la raíz del proyecto.")
    sys.exit(1)

# ==========================================
# 1. CLASE WRAPPER (Hailo -> Dict(NCHW))
# ==========================================
class PillarNest_Hailo_Module(torch.nn.Module):
    def __init__(self, runner, context_type):
        super().__init__()
        self._runner = runner
        self.headers = ['reg', 'height', 'dim', 'rot', 'vel', 'iou', 'heatmap']
        print(f"⚙️  Contexto: {context_type}")
        with runner.infer_context(context_type) as ctx:
            self._hailo_model = runner.get_keras_model(ctx)

    def forward(self, input_tensor):        
        # PyTorch NCHW -> Hailo NHWC
        input_nhwc = np.transpose(input_tensor.cpu().detach().numpy(), (0, 2, 3, 1))
        
        try: 
            raw_outputs = self._hailo_model({'input_pillars': input_nhwc})
        except: 
            raw_outputs = self._hailo_model(input_nhwc)

        # Recuperación blindada de nombres
        if isinstance(raw_outputs, list):
            keys = None
            try: keys = list(self._runner.get_hn_model().get_end_node_names())
            except: pass
            
            if not keys: keys = getattr(self._hailo_model, 'output_names', None)
            
            if (not keys) and hasattr(self._hailo_model, 'outputs') and self._hailo_model.outputs:
                 keys = [t.name.split(':')[0].split('/')[0] if hasattr(t, 'name') else str(t) for t in self._hailo_model.outputs]
                 
            if keys is None or len(keys) != len(raw_outputs):
                keys = [f"task{i}_{h}" for i in range(6) for h in self.headers]
                if len(keys) != len(raw_outputs): keys = [f"out_{i}" for i in range(len(raw_outputs))]
                
            raw_outputs = dict(zip(keys, raw_outputs))

        # Convertir a NCHW (Formato PyTorch esperado por PostProcessor)
        clean_outputs = {}
        for k, v in raw_outputs.items():
            if hasattr(v, 'numpy'): v = v.numpy()
            else: v = np.array(v)
            
            clean_k = k.split(':')[0].split('/')[0]
            if v.ndim == 4: 
                v = v.transpose(0, 3, 1, 2) # NHWC -> NCHW
            clean_outputs[clean_k] = v
            
        return clean_outputs

# ==========================================
# 2. FUNCIONES DE VISUALIZACIÓN Y DECODE
# ==========================================
def get_corners(x, y, w, l, yaw):
    """Calcula las 4 esquinas del bounding box rotado"""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    rotated = np.dot(corners, R.T) + np.array([x, y])
    return rotated

def draw_boxes(ax, dict_boxes, color, label, linestyle='-'):
    """ Dibuja las cajas procesando la nueva estructura de diccionarios """
    count = 0
    for b in dict_boxes:
        # El postprocesador ya filtra por score_threshold internamente, 
        # pero aplicamos un filtro extra visual si queremos destacar los más seguros
        if b['score'] < 0.25: continue
        count += 1
        
        box = b['box']
        x, y = box['x'], box['y']
        w, l = box['w'], box['l']
        rot = box['rot']
        
        # FIX VISUAL W/L típico de NuScenes/CenterPoint
        l, w = w, l 
        
        corners = get_corners(x, y, w, l, rot)
        poly = patches.Polygon(corners, closed=True, edgecolor=color, facecolor='none', linewidth=1.5, linestyle=linestyle)
        ax.add_patch(poly)
        
        # Dibujar línea de dirección (Heading)
        center = np.array([x, y])
        head = center + np.array([np.cos(rot), np.sin(rot)]) * (l/2)
        ax.plot([center[0], head[0]], [center[1], head[1]], color=color, linewidth=1)
    
    return patches.Patch(color=color, label=f"{label} ({count})")

def decode_outputs_with_postprocessor(output_dict, postprocessor):
    """ 
    Alinea el diccionario de salidas al formato de lista plana que exige el 
    nuevo CenterPointPostProcessor y ejecuta el forward.
    """
    flat_outputs = []
    
    # El postprocesador espera estrictamente este orden: Tarea0(reg..heatmap), Tarea1(reg..heatmap)...
    for i in range(6):
        for h in postprocessor.headers:
            key = f"task{i}_{h}"
            if key in output_dict:
                # Aseguramos que tenga batch dimension
                tensor = output_dict[key]
                if tensor.ndim == 3: 
                    tensor = tensor[np.newaxis, ...]
                flat_outputs.append(tensor)
            else:
                print(f"⚠️ Alerta: Tensor {key} no encontrado. Inyectando ceros.")
                flat_outputs.append(np.zeros((1, 1, 1, 1), dtype=np.float32))
                
    # Ejecutamos tu lógica oficial
    final_results = postprocessor.forward(flat_outputs)
    return final_results

# ==========================================
# 3. MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--har', required=True, help="Modelo FP32")
    parser.add_argument('--qhar', required=True, help="Modelo INT8")
    parser.add_argument('--input', required=True, help="Input .npy (NCHW)")
    parser.add_argument('--golden', required=True, help="Golden data .npz")
    parser.add_argument('--bin', required=True, help="PCD .bin para fondo")
    args = parser.parse_args()

    print("⚙️  Inicializando Post-Procesador...")
    postprocessor = CenterPointPostProcessor(Cfg)

    print("📂 Cargando datos...")
    input_data = np.load(args.input)
    input_tensor = torch.from_numpy(input_data)
    
    # Golden data puede ser npz. Lo convertimos a dict estándar.
    golden_data_raw = np.load(args.golden)
    golden_data = {k: golden_data_raw[k] for k in golden_data_raw.files}
    
    try: points = np.fromfile(args.bin, dtype=np.float32).reshape(-1, 5)
    except: points = np.fromfile(args.bin, dtype=np.float32).reshape(-1, 4)

    # --- INFERENCIAS ---
    print("\n🔵 Ejecutando FP32 (Emulador)...")
    boxes_fp = []
    try:
        runner_fp = ClientRunner(har=args.har)
        try: runner_fp._sdk_backend.update_fp_model(runner_fp._sdk_backend.model)
        except: pass
        mod_fp = PillarNest_Hailo_Module(runner_fp, InferenceContext.SDK_FP_OPTIMIZED)
        fp_dict = mod_fp(input_tensor)
        boxes_fp = decode_outputs_with_postprocessor(fp_dict, postprocessor)
    except Exception as e: print(f"   ⚠️ Falló FP32: {e}")

    print("\n🔴 Ejecutando INT8 (Emulador)...")
    boxes_q = []
    try:
        runner_q = ClientRunner(har=args.qhar)
        mod_q = PillarNest_Hailo_Module(runner_q, InferenceContext.SDK_QUANTIZED)
        q_dict = mod_q(input_tensor)
        boxes_q = decode_outputs_with_postprocessor(q_dict, postprocessor)
    except Exception as e: print(f"   ⚠️ Falló INT8: {e}")

    print("\n🟢 Procesando Golden (PyTorch)...")
    boxes_golden = decode_outputs_with_postprocessor(golden_data, postprocessor)

    # ==========================================
    # --- PLOT PARALELO (SIDE-BY-SIDE) ---
    # ==========================================
    print(f"\n📊 Generando gráfico paralelo...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='black')
    
    scenarios = [
        {"ax": axes[0], "boxes": boxes_fp,     "color": 'cyan', "title": "FP32 (Emulador SDK)", "style": '-'},
        {"ax": axes[1], "boxes": boxes_q,      "color": 'red',  "title": "INT8 (Quantized)",     "style": '-'},
        {"ax": axes[2], "boxes": boxes_golden, "color": 'lime', "title": "Golden (PyTorch)",    "style": '--'}
    ]

    for config in scenarios:
        ax = config["ax"]
        boxes = config["boxes"]
        color = config["color"]
        title = config["title"]
        style = config["style"]

        ax.set_facecolor('black')
        
        # Nube de puntos de fondo
        ax.scatter(points[::5, 0], points[::5, 1], s=0.1, c='gray', alpha=0.5)
        
        handles = []
        if len(boxes) > 0:
            h = draw_boxes(ax, boxes, color, title.split()[0], style)
            handles.append(h)
            ax.legend(handles=handles, loc='upper right', facecolor='black', labelcolor='white')
        else:
             ax.text(0, 0, "Sin Detecciones", color='white', ha='center')

        # Configuración común de límites (-60m a +60m en X e Y)
        ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
        ax.set_title(title, color='white', fontsize=14)
        ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')

    # Guardado de la imagen
    frame_id = os.path.basename(args.bin)
    fig.suptitle(f"Comparativa Paralela PillarNest | Frame: {frame_id}", color='white', fontsize=16, y=0.99)
    plt.tight_layout()
    
    out_file = "debug/parallel_comparison_multisweep3.png"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"🖼️  Imagen paralela guardada en: {out_file}")

if __name__ == "__main__":
    main()
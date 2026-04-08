import argparse
import numpy as np
import torch
import sys
import os
from tqdm import tqdm
from hailo_sdk_client import ClientRunner, InferenceContext

headers = ['reg', 'height', 'dim', 'rot', 'vel', 'iou', 'heatmap']

# ==========================================
# 1. MÉTRICAS CIENTÍFICAS (PTQ SOTA)
# ==========================================
def calculate_sqnr(fp32_tensor, int8_tensor):
    """Calcula el Signal-to-Quantization-Noise Ratio (SQNR) en dB."""
    sig = fp32_tensor.flatten()
    noise = (fp32_tensor - int8_tensor).flatten()
    
    sig_power = np.sum(sig ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf') 
    
    sqnr = 10 * np.log10(sig_power / (noise_power + 1e-9))
    return sqnr

def calculate_cosine_similarity(fp32_tensor, int8_tensor):
    """Calcula la similitud del coseno (ideal para heatmaps)."""
    a = fp32_tensor.flatten()
    b = int8_tensor.flatten()
    
    dot_product = np.sum(a * b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b + 1e-9)

# ==========================================
# 2. CLASE WRAPPER (Hailo -> PyTorch)
# ==========================================
class PillarNest_Hailo_Module(torch.nn.Module):
    def __init__(self, runner):
        super().__init__()
        # Activamos el modo cuantizado (bit-accurate simulation)
        print(f"⚙️  Iniciando contexto de inferencia: SDK_QUANTIZED")
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
            self._hailo_model = runner.get_keras_model(ctx)

    def forward(self, input_tensor):        
        # Transponer NCHW -> NHWC
        input_nhwc = np.transpose(input_tensor.cpu().detach().numpy(), (0, 2, 3, 1))

        raw_outputs = self._hailo_model(input_nhwc)

        # Recuperación de nombres
        if isinstance(raw_outputs, list):
            keys = []
            for i in range(6):
                for h in headers:
                    keys.append(f"task{i}_{h}")
            if len(keys) == len(raw_outputs):
                raw_outputs = dict(zip(keys, raw_outputs))

        flat_outputs = []
        
        for i in range(6):
            for h in headers:
                target_key = f"task{i}_{h}"
                found = None
                
                if target_key in raw_outputs: found = raw_outputs[target_key]
                if found is None:
                    for k in raw_outputs:
                        if target_key in k: found = raw_outputs[k]; break
                
                if found is not None:
                    arr = np.array(found)
                    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
                    flat_outputs.append(t)
                else:
                    flat_outputs.append(torch.zeros(1, 1, 1, 1))
                    
        return tuple(flat_outputs)

# ==========================================
# 3. FUNCIÓN PRINCIPAL DE VERIFICACIÓN
# ==========================================
def verify_quantized(qhar_path, input_npy, golden_npz):
    print(f"🔌 Cargando Q-HAR: {qhar_path}")
    runner = ClientRunner(har=qhar_path)
    
    print("🚀 Levantando Emulador INT8...")
    hailo_net = PillarNest_Hailo_Module(runner)
    
    print("📂 Cargando Golden Data...")
    try:
        input_data = np.load(input_npy)
        golden_data = np.load(golden_npz)
        
        # --- AÑADE ESTO PARA LIMITAR LOS FRAMES ---
        limit_frames = 128  # Cambia este número a 16 o los que quieras probar
        print(f"✂️  Limitando verificación a los primeros {limit_frames} frames para ahorrar tiempo.")
        input_data = input_data[:limit_frames]
        
        # También debemos recortar los datos "Golden" para que las métricas coincidan
        new_golden = {}
        for key in golden_data.keys():
            new_golden[key] = golden_data[key][:limit_frames]
        golden_data = new_golden
        # ------------------------------------------
        
    except Exception as e:
        print(f"❌ Error leyendo archivos numpy: {e}")
        return
        
    num_frames = input_data.shape[0]
    print(f"📦 Verificando {num_frames} frames.")

    expected_names = []
    for i in range(6):
        for h in headers:
            expected_names.append(f"task{i}_{h}")

    # Acumulador para los frames emulados
    accumulated_outputs = {name: [] for name in expected_names}

    print("🏃 Ejecutando inferencia INT8 frame a frame...")
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Emulando Q-HAR"):
            single_input = torch.from_numpy(input_data[i:i+1])
            outputs_tuple = hailo_net(single_input)
            
            for j, name in enumerate(expected_names):
                accumulated_outputs[name].append(outputs_tuple[j].numpy())

    print("\n" + "="*80)
    print("📊 REPORTE DE DEGRADACIÓN DE CUANTIZACIÓN (PTQ SOTA)")
    print("="*80)

    # --- ACUMULADORES PARA EL RESUMEN EJECUTIVO ---
    sum_sqnr = 0.0
    count_sqnr = 0
    sum_cosine = 0.0
    count_cosine = 0
    
    degraded_sqnr_count = 0
    degraded_cosine_count = 0
    
    sum_geom_mae = 0.0
    count_geom_mae = 0

    # Imprimimos agrupado por Task
    for task_idx in range(6):
        print(f"\n🎯 --- TASK {task_idx} ---")
        print(f"{'CAPA':<15} | {'MÉTRICA':<15} | {'VALOR':<10} | {'STATUS'}")
        print("-" * 60)
        
        for h in headers:
            name = f"task{task_idx}_{h}"
            
            if name not in golden_data:
                continue
                
            gt_tensor = golden_data[name]
            hailo_tensor = np.concatenate(accumulated_outputs[name], axis=0)
            
            if hailo_tensor.shape != gt_tensor.shape:
                print(f"{name:<15} | ❌ ERROR DE SHAPE")
                continue

            # Selección de Métrica y Umbral (Threshold)
            if "heatmap" in h:
                metric_name = "Cosine Sim."
                val = calculate_cosine_similarity(gt_tensor, hailo_tensor)
                
                sum_cosine += val
                count_cosine += 1
                
                # Un heatmap > 0.95 preserva su forma
                if val > 0.95:
                    status = "✅ OK"
                else:
                    status = "⚠️ DEGRADADO"
                    degraded_cosine_count += 1
                    
                print(f"{h.upper():<15} | {metric_name:<15} | {val:>8.4f}   | {status}")
                
            else:
                metric_name = "SQNR (dB)"
                val = calculate_sqnr(gt_tensor, hailo_tensor)
                
                sum_sqnr += val
                count_sqnr += 1
                
                # Regresiones geométricas > 15 dB son seguras
                if val > 15.0:
                    status = "✅ OK"
                else:
                    status = "⚠️ RUIDO ALTO"
                    degraded_sqnr_count += 1
                
                # IDEA 3: Calculamos el MAE solo para geometría
                mae = np.abs(gt_tensor - hailo_tensor).mean()
                sum_geom_mae += mae
                count_geom_mae += 1
                
                print(f"{h.upper():<15} | {metric_name:<15} | {val:>8.2f}   | {status}")

    # ====================================================================
    # RESUMEN EJECUTIVO (Tus 3 ideas combinadas)
    # ====================================================================
    avg_sqnr = sum_sqnr / count_sqnr if count_sqnr > 0 else 0
    avg_cosine = sum_cosine / count_cosine if count_cosine > 0 else 0
    avg_geom_mae = sum_geom_mae / count_geom_mae if count_geom_mae > 0 else 0

    print("\n" + "="*80)
    print("🚀 RESUMEN EJECUTIVO DE OPTIMIZACIÓN (PTQ)")
    print("="*80)
    
    print("📌 1. Medias Globales (SOTA)")
    print(f"  - Promedio Global SQNR (Geometría):  {avg_sqnr:>6.2f} dB  (Buscamos > 15 dB)")
    print(f"  - Promedio Global Coseno (Heatmaps): {avg_cosine:>6.4f}     (Buscamos > 0.95)")
    
    print("\n🚥 2. Semáforos Rojos (Capas Degradadas)")
    print(f"  - 🚨 Geometría en Peligro (SQNR < 15 dB):  {degraded_sqnr_count:2d} / {count_sqnr}")
    print(f"  - 🚨 Heatmaps en Peligro (Coseno < 0.95):  {degraded_cosine_count:2d} / {count_cosine}")
    
    print("\n📏 3. Error Absoluto (Geometría pura)")
    print(f"  - MAE Geométrico Global: {avg_geom_mae:.6f} (Mientras más cercano a 0, mejor)")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('qhar', help='Ruta al .q.har generado')
    parser.add_argument('--input', help='Ruta al golden_inputs.npy')
    parser.add_argument('--golden', help='Ruta al golden_outputs.npz')
    
    args = parser.parse_args()
    verify_quantized(args.qhar, args.input, args.golden)
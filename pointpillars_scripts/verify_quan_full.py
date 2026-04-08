import argparse
import numpy as np
import torch
import sys
import os
from tqdm import tqdm
from hailo_sdk_client import ClientRunner, InferenceContext

# ==========================================
# 1. MÉTRICAS SOTA (VALORACIÓN DE DEGRADACIÓN)
# ==========================================
def calculate_sqnr(fp32_tensor, int8_tensor):
    """Calcula el Signal-to-Quantization-Noise Ratio (SQNR) en dB."""
    sig = fp32_tensor.flatten()
    noise = (fp32_tensor - int8_tensor).flatten()
    sig_power = np.sum(sig ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power == 0: return float('inf') 
    return 10 * np.log10(sig_power / (noise_power + 1e-9))

def calculate_cosine_similarity(fp32_tensor, int8_tensor):
    """Calcula la similitud del coseno (ideal para cls_score)."""
    a, b = fp32_tensor.flatten(), int8_tensor.flatten()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.sum(a * b) / (norm_a * norm_b + 1e-9)

# ==========================================
# 2. CLASE WRAPPER (PointPillars INT8 Emulator)
# ==========================================
PP_OUTPUTS = ['cls_score', 'bbox_pred', 'dir_cls_pred']

class PointPillars_Hailo_Quant_Module(torch.nn.Module):
    def __init__(self, runner):
        super().__init__()
        # Activamos la simulación exacta de hardware (Bit-accurate)
        print(f"⚙️  Iniciando Emulador INT8: SDK_QUANTIZED")
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
            self._hailo_model = runner.get_keras_model(ctx)

    def forward(self, input_tensor):        
        # NCHW -> NHWC
        input_nhwc = np.transpose(input_tensor.cpu().detach().numpy(), (0, 2, 3, 1))
        
        try:
            raw_outputs = self._hailo_model({'input_canvas': input_nhwc})
        except:
            raw_outputs = self._hailo_model(input_nhwc)

        # Mapeo de salidas a diccionario
        if isinstance(raw_outputs, list):
            # En modo cuantizado, a veces los nombres cambian ligeramente
            # pero el orden se mantiene respecto al HAR
            raw_outputs = dict(zip(PP_OUTPUTS, raw_outputs))

        processed_outputs = {}
        for target_key in PP_OUTPUTS:
            found = None
            if target_key in raw_outputs: 
                found = raw_outputs[target_key]
            else:
                # Búsqueda por subcadena para mayor robustez
                for k in raw_outputs:
                    if target_key in k: found = raw_outputs[k]; break
            
            if found is not None:
                arr = np.array(found)
                # NHWC -> NCHW para comparar con Golden PyTorch
                processed_outputs[target_key] = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        return processed_outputs

# ==========================================
# 3. FUNCIÓN DE VERIFICACIÓN PRINCIPAL
# ==========================================
def verify_pointpillars_quantized(qhar_path, input_npy, golden_npz, limit_frames=32):
    print(f"🔌 Cargando Q-HAR: {qhar_path}")
    runner = ClientRunner(har=qhar_path)
    hailo_net = PointPillars_Hailo_Quant_Module(runner)
    
    print("📂 Cargando Golden Data...")
    try:
        input_data = np.load(input_npy)[:limit_frames]
        golden_data = np.load(golden_npz)
        print(f"✂️  Verificando {limit_frames} frames (Muestreo para agilizar).")
    except Exception as e:
        print(f"❌ Error leyendo archivos: {e}"); return

    # Acumuladores
    acc_outputs = {name: [] for name in PP_OUTPUTS}

    with torch.no_grad():
        for i in tqdm(range(input_data.shape[0]), desc="Emulando INT8"):
            single_input = torch.from_numpy(input_data[i:i+1])
            outputs_dict = hailo_net(single_input)
            for k, v in outputs_dict.items():
                if v is not None: acc_outputs[k].append(v.numpy())

    # # =========================================================================
    # # 🚨 NUEVO BLOQUE DE DIAGNÓSTICO DE SHAPES (INSERTA ESTO AQUÍ)
    # # =========================================================================
    # print("\n" + "="*85)
    # print("🔍 DIAGNÓSTICO DE FORMAS (SHAPES): ¿ESTÁN CRUZADAS LAS CABEZAS?")
    # print("="*85)
    # for name in PP_OUTPUTS:
    #     if name in golden_data and acc_outputs[name]:
    #         gt_shape = golden_data[name][:limit_frames].shape
    #         hailo_shape = np.concatenate(acc_outputs[name], axis=0).shape
    #         print(f"CABEZA: {name}")
    #         print(f"  -> PyTorch (Golden) Shape : {gt_shape}")
    #         print(f"  -> Hailo (INT8) Shape     : {hailo_shape}")
            
    #         # Verificamos si los canales (dimensión 1) coinciden
    #         if gt_shape[1] != hailo_shape[1]:
    #             print(f"  -> 🚨 ¡ALERTA ROJA! Los canales no coinciden. ¡TENSORES CRUZADOS!")
    #         else:
    #             print(f"  -> ✅ Shapes idénticos")
    #         print("-" * 60)
    # # =========================================================================
    
    
    print("\n" + "="*85)
    print(f"📊 REPORTE DE DEGRADACIÓN DE CUANTIZACIÓN (POINTPILLARS)")
    print("="*85)
    print(f"{'CABEZA DE SALIDA':<20} | {'MÉTRICA':<15} | {'VALOR':<10} | {'STATUS'}")
    print("-" * 85)

    stats = {'sqnr': [], 'cosine': [], 'mae': []}

    for name in PP_OUTPUTS:
        if name not in golden_data or not acc_outputs[name]:
            print(f"{name:<20} | ❌ MISSING")
            continue
            
        gt = golden_data[name][:limit_frames]
        hailo = np.concatenate(acc_outputs[name], axis=0)
        
        # 1. MAE (General)
        mae = np.abs(gt - hailo).mean()
        stats['mae'].append(mae)

        # 2. Selección de métrica según el tipo de salida
        if "score" in name or "dir" in name:
            metric_label = "Cosine Sim."
            val = calculate_cosine_similarity(gt, hailo)
            stats['cosine'].append(val)
            status = "✅ OK" if val > 0.95 else "⚠️ DEGRADADO"
            print(f"{name:<20} | {metric_label:<15} | {val:>8.4f}   | {status}")
        else:
            metric_label = "SQNR (dB)"
            val = calculate_sqnr(gt, hailo)
            stats['sqnr'].append(val)
            status = "✅ OK" if val > 15.0 else "⚠️ RUIDO ALTO"
            print(f"{name:<20} | {metric_label:<15} | {val:>8.2f}   | {status}")

    # --- RESUMEN EJECUTIVO ---
    print("\n" + "="*85)
    print("🚀 RESUMEN EJECUTIVO DE CUANTIZACIÓN")
    print("="*85)
    print(f"  - SQNR Promedio (Regresión): {np.mean(stats['sqnr']):>6.2f} dB  (Ideal > 15dB)")
    print(f"  - Coseno Promedio (Clases):   {np.mean(stats['cosine']):>6.4f}     (Ideal > 0.95)")
    print(f"  - MAE Geométrico Global:     {np.mean(stats['mae']):.6f}")
    print("="*85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('qhar', help='Archivo .q.har')
    parser.add_argument('--input', required=True, help='Inputs .npy')
    parser.add_argument('--golden', required=True, help='Outputs .npz PyTorch')
    parser.add_argument('--limit', type=int, default=32, help='Frames a verificar')
    
    args = parser.parse_args()
    verify_pointpillars_quantized(args.qhar, args.input, args.golden, args.limit)
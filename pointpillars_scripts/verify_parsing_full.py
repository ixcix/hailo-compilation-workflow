import argparse
import numpy as np
import torch
import sys
from tqdm import tqdm
from hailo_sdk_client import ClientRunner, InferenceContext

# ==========================================
# 1. CLASE WRAPPER (PointPillars Hailo)
# ==========================================
# Definimos las 3 salidas oficiales de PointPillars
PP_OUTPUTS = ['cls_score', 'bbox_pred', 'dir_cls_pred']

class PointPillars_Hailo_Module(torch.nn.Module):
    """ 
    Wrapper para el Emulador de Hailo adaptado a PointPillars.
    """
    def __init__(self, runner, emulate_native=True):
        super().__init__()
        self._runner = runner
        
        # Para verificar el parsing, siempre usamos SDK_NATIVE (FP32)
        context_type = InferenceContext.SDK_NATIVE  
        print(f"⚙️  Iniciando Emulador en contexto: {context_type}")

        with runner.infer_context(context_type) as ctx:
            self._hailo_model = runner.get_keras_model(ctx)
            if self._hailo_model is None:
                raise ValueError("❌ El modelo Hailo no se cargó correctamente.")

    def forward(self, input_tensor):        
        # 1. PyTorch NCHW -> Hailo NHWC
        input_nhwc = np.transpose(input_tensor.cpu().detach().numpy(), (0, 2, 3, 1))

        # 2. Inferencia en el Emulador
        # Intentamos usar el nombre del nodo de entrada definido en el parsing
        try:
            raw_outputs = self._hailo_model({'input_canvas': input_nhwc})
        except:
            # Fallback si el nombre es distinto
            raw_outputs = self._hailo_model(input_nhwc)

        # 3. MAPEADO DE SALIDAS (Específico PointPillars)
        # El emulador devuelve una lista o un dict dependiendo de la versión
        if isinstance(raw_outputs, list):
            # Intentamos recuperar los nombres reales del HAR
            try:
                keys = list(self._runner.get_hn_model().get_end_node_names())
            except:
                keys = PP_OUTPUTS
            raw_outputs = dict(zip(keys, raw_outputs))

        processed_outputs = {}
        for target_key in PP_OUTPUTS:
            found_tensor = None
            # Buscamos coincidencia exacta o parcial (por prefijos de Keras/Hailo)
            for k in raw_outputs.keys():
                if target_key in k:
                    found_tensor = raw_outputs[k]
                    break
            
            if found_tensor is not None:
                # Hailo NHWC -> PyTorch NCHW para comparar con el Golden
                arr = found_tensor.numpy() if hasattr(found_tensor, 'numpy') else np.array(found_tensor)
                processed_outputs[target_key] = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            else:
                print(f"⚠️ Alerta: No se encontró la salida {target_key} en el emulador.")
                processed_outputs[target_key] = None

        return processed_outputs

# ==========================================
# 2. FUNCIÓN DE VERIFICACIÓN
# ==========================================
def verify_pointpillars_har(har_path, input_npy, golden_npz):
    print(f"🔌 Cargando HAR: {har_path}")
    runner = ClientRunner(har=har_path)
    
    hailo_net = PointPillars_Hailo_Module(runner)
    
    print("📂 Cargando Golden Tensors...")
    try:
        # Tensors generados previamente con el script de debug de PyTorch
        input_data = np.load(input_npy) 
        golden_data = np.load(golden_npz)
    except Exception as e:
        print(f"❌ Error leyendo archivos: {e}")
        return
    
    num_frames = input_data.shape[0]
    print(f"📦 Verificando {num_frames} frames...")

    # Acumuladores para el reporte final
    accumulated_results = {k: [] for k in PP_OUTPUTS}

    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Emulando"):
            # (1, C, H, W)
            single_input = torch.from_numpy(input_data[i:i+1])
            outputs_dict = hailo_net(single_input)
            
            for k, v in outputs_dict.items():
                if v is not None:
                    accumulated_results[k].append(v.numpy())

    print("\n⚖️  COMPARATIVA FP32: HAILO EMULATOR VS PYTORCH GOLDEN")
    print("-" * 95)
    print(f"{'CABEZA DE SALIDA':<20} | {'STATUS':<10} | {'ERROR (MAE)':<15} | {'ERROR MAX (MAX-E)'}")
    print("-" * 95)

    total_mae = 0
    valid_count = 0
    
    for name in PP_OUTPUTS:
        if name in golden_data and len(accumulated_results[name]) > 0:
            hailo_tensor = np.concatenate(accumulated_results[name], axis=0)
            gt_tensor = golden_data[name]
            
            if hailo_tensor.shape != gt_tensor.shape:
                print(f"{name:<20} | ❌ SHAPE  | Got {hailo_tensor.shape} vs {gt_tensor.shape}")
                continue

            # Cálculo de discrepancias detallado
            abs_diff = np.abs(gt_tensor - hailo_tensor)
            mae = abs_diff.mean()
            max_e = abs_diff.max()
            
            # Buscamos las coordenadas exactas donde ocurre el error máximo
            max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            
            total_mae += mae
            valid_count += 1
            
            status = "✅ OK" if mae < 1e-5 else "⚠️ HIGH"
            print(f"{name:<20} | {status} | {mae:.8e}     | {max_e:.8e}")
            print(f"   ↳ 🎯 Coords del Max Error (Batch, Channel, Y, X): {max_idx}")

            total_mae += mae
            valid_count += 1
            
            # Tolerancia estricta para FP32
            status = "✅ OK" if mae < 1e-5 else "⚠️ HIGH"
            print(f"{name:<20} | {status} | {mae:.8e}     | {max_e:.8e}")
        else:
            print(f"{name:<20} | ❌ MISSING | No se pudo comparar esta capa.")

    print("-" * 95)
    if valid_count > 0:
        avg_error = total_mae / valid_count
        print(f"🏁 Error Global Promedio: {avg_error:.8e}")
        if avg_error < 1e-5:
            print("🏆 [VERIFICACIÓN EXITOSA] El modelo es funcionalmente idéntico a PyTorch.")
        else:
            print("❌ [AVISO] Se detectan discrepancias significativas. Revisa el parsing.")
    else:
        print("❌ No se pudo realizar la comparación.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('har', help='Archivo .har generado tras el parsing')
    parser.add_argument('--input', required=True, help='Archivo .npy con los inputs (canvas)')
    parser.add_argument('--golden', required=True, help='Archivo .npz con los outputs de PyTorch')
    
    args = parser.parse_args()
    verify_pointpillars_har(args.har, args.input, args.golden)
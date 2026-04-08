import argparse
import numpy as np
import torch
import sys
from tqdm import tqdm
from hailo_sdk_client import ClientRunner, InferenceContext

# ==========================================
# 1. CLASE WRAPPER (Hailo -> PyTorch)
# ==========================================
headers = ['reg', 'height', 'dim', 'rot', 'vel', 'iou', 'heatmap']

class PillarNest_Hailo_Module(torch.nn.Module):
    """ 
    Reemplazo para la red completa.
    Ejecuta el emulador de Hailo y formatea la salida como espera mmdetection3d.
    """
    def __init__(self, runner, emulate_quantized=False, use_hw=False, emulate_native=False):
        super().__init__()
        self._runner = runner
        
        # Selección de contexto
        if use_hw:
            context_type = InferenceContext.SDK_HAILO_HW
        elif emulate_quantized:
            context_type = InferenceContext.SDK_QUANTIZED 
        elif emulate_native:
            context_type = InferenceContext.SDK_NATIVE
        else:
            context_type = InferenceContext.SDK_FP_OPTIMIZED

        print(f"⚙️  Contexto de inferencia seleccionado: {context_type}")

        with runner.infer_context(context_type) as ctx:
            self._hailo_model = runner.get_keras_model(ctx)
            if self._hailo_model is None:
                raise ValueError("❌ El modelo Hailo no se cargó correctamente.")

    def forward(self, input_tensor):        
        # 1. Transponer a NHWC (PyTorch NCHW -> Hailo NHWC)
        input_nhwc = np.transpose(input_tensor.cpu().detach().numpy(), (0, 2, 3, 1))

        # 2. Inferencia en el Emulador
        try:
            raw_outputs = self._hailo_model({'input_pillars': input_nhwc})
        except:
            raw_outputs = self._hailo_model(input_nhwc)

        # 3. RECUPERACIÓN DE NOMBRES DE SALIDA (BLINDADO)
        if isinstance(raw_outputs, list):
            keys = None
            try:
                keys = self._runner.get_hn_model().get_end_node_names()
                if keys: keys = list(keys)
            except:
                pass

            if not keys:
                keys = getattr(self._hailo_model, 'output_names', None)
            
            if (not keys) and hasattr(self._hailo_model, 'outputs'):
                if self._hailo_model.outputs is not None:
                    keys = [t.name.split(':')[0].split('/')[0] if hasattr(t, 'name') else str(t) for t in self._hailo_model.outputs]

            if keys is None or len(keys) != len(raw_outputs):
                keys = [f"task{i}_{h}" for i in range(6) for h in headers]
                if len(keys) != len(raw_outputs):
                    keys = [f"out_{i}" for i in range(len(raw_outputs))]

            raw_outputs = dict(zip(keys, raw_outputs))

        flat_outputs = {}
        
        # Recorremos 6 tareas x 7 cabeceras = 42 tensores
        for i in range(6):
            for h in headers:
                target_key = f"task{i}_{h}"
                found_tensor = None
                
                if target_key in raw_outputs:
                    found_tensor = raw_outputs[target_key]
                
                if found_tensor is None:
                    for key in raw_outputs.keys():
                        if target_key == key or key.startswith(f"{target_key}/"):
                            found_tensor = raw_outputs[key]
                            break
                
                if found_tensor is not None:
                    arr = found_tensor.numpy() if hasattr(found_tensor, 'numpy') else np.array(found_tensor)
                    out_tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
                    flat_outputs[target_key] = out_tensor
                else:
                    flat_outputs[target_key] = torch.zeros(1, 1, 1, 1)

        return flat_outputs

# ==========================================
# 2. FUNCIÓN DE VERIFICACIÓN (Adaptada a N Frames)
# ==========================================
def verify_har(har_path, input_npy, golden_npz):
    print(f"🔌 Cargando HAR: {har_path}")
    runner = ClientRunner(har=har_path)
    
    # 🚨 ASEGURAMOS EVALUACIÓN EN FP32 (Native)
    print("🚀 Iniciando Emulador (FP32 NATIVE)...")
    hailo_net = PillarNest_Hailo_Module(runner, emulate_native=True)
    
    print("📂 Cargando Golden Data...")
    try:
        input_data = np.load(input_npy) # Shape esperado: (128, 48, 720, 720)
        golden_data = np.load(golden_npz)
    except Exception as e:
        print(f"❌ Error leyendo archivos numpy: {e}")
        return
    
    num_frames = input_data.shape[0]
    print(f"📦 Se encontraron {num_frames} frames para verificar.")

    # Diccionario para acumular las salidas a lo largo de los frames
    accumulated_outputs = {f"task{i}_{h}": [] for i in range(6) for h in headers}

    print("🏃 Ejecutando forward pass frame a frame...")
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Emulando FP32"):
            # Extraer un frame: shape (1, 48, 720, 720)
            single_input = torch.from_numpy(input_data[i:i+1])
            
            # Inferencia de un frame
            outputs_dict = hailo_net(single_input)
            
            # Acumular
            for k, v in outputs_dict.items():
                if k in accumulated_outputs:
                    accumulated_outputs[k].append(v.numpy())

    print("\n⚖️  RESULTADOS DE LA COMPARACIÓN FP32 (Hailo Native vs PyTorch):")
    print("-" * 80)
    print(f"{'CAPA':<20} | {'STATUS':<10} | {'ERROR (MAE)':<15} | {'MAX VAL'}")
    print("-" * 80)

    total_error = 0
    count = 0
    
    for name in accumulated_outputs.keys():
        if name in golden_data:
            # Juntamos la lista de (1, C, H, W) en (128, C, H, W)
            hailo_tensor = np.concatenate(accumulated_outputs[name], axis=0)
            gt_tensor = golden_data[name]
            
            if hailo_tensor.shape != gt_tensor.shape:
                print(f"{name:<20} | ❌ SHAPE  | Got {hailo_tensor.shape} vs {gt_tensor.shape}")
                continue

            # Cálculo de Error Absoluto Medio (MAE) sobre TODOS los frames
            diff = np.abs(gt_tensor - hailo_tensor).mean()
            max_val = np.max(np.abs(gt_tensor))
            total_error += diff
            count += 1
            
            # Tolerancia para FP32 (Debería ser pequeñísimo, error de coma flotante)
            status = "✅ OK" if diff < 1e-4 else "⚠️ HIGH"
            print(f"{name:<20} | {status} | {diff:.8f}        | {max_val:.4f}")
        else:
            print(f"{name:<20} | ❌ NO GT   | N/A")

    print("-" * 80)
    if count > 0:
        print(f"🏁 Error Global Promedio (FP32): {total_error/count:.8e}")
        if (total_error/count) < 1e-4:
            print("🏆 El parseo del modelo es PERFECTO. Puedes proceder a cuantizar con seguridad.")
        else:
            print("⚠️ Hay discrepancias en FP32. Revisa la exportación a ONNX/Hailo.")
    else:
        print("⚠️ No se pudo comparar nada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('har', help='Ruta al archivo .har (Pre-cuantización preferiblemente)')
    parser.add_argument('--input', required=True,  help='Ruta al golden_inputs.npy')
    parser.add_argument('--golden', required=True, help='Ruta al golden_outputs.npz')
    
    args = parser.parse_args()
    verify_har(args.har, args.input, args.golden)
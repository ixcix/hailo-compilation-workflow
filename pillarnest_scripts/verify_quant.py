import argparse
import numpy as np
import torch
import sys
import os
from hailo_sdk_client import ClientRunner, InferenceContext


headers = ['reg', 'height', 'dim', 'rot', 'vel', 'iou', 'heatmap']

# headers = ['reg', 'height', 'dim', 'rot', 'iou', 'heatmap']

# --- CLASE WRAPPER (Igual que antes pero activando Quantized) ---
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
        # Inferencia
        # Nota: Keras espera un dict o array. Probamos dict primero.
        try:
            # Intentamos auto-detectar nombre si es posible, si no, array directo
            raw_outputs = self._hailo_model(input_nhwc) 
            print(f"✅ Inferencia exitosa con input directo.")
        except:
            # Fallback
            raw_outputs = self._hailo_model({'input_layer1': input_nhwc})
            print(f"⚠️ Fallback a dict con 'input_layer1'. Verifica el nombre de tu input en el HAR.")


        print(f"🔍 Tipo de salida del modelo: {type(raw_outputs)}"
              f" | Claves: {list(raw_outputs.keys()) if isinstance(raw_outputs, dict) else 'N/A'}"  )
        # print(raw_outputs[0])
        # exit()
        # Recuperación de nombres (Asumimos el orden verificado anteriormente)
        # Si raw_outputs es lista, la mapeamos.
        if isinstance(raw_outputs, list):
            keys = []
            for i in range(6):
                for h in headers:
                    keys.append(f"task{i}_{h}")
            
            # Si coinciden longitudes, creamos dict
            if len(keys) == len(raw_outputs):
                raw_outputs = dict(zip(keys, raw_outputs))

        flat_outputs = []
        
        
        for i in range(6):
            for h in headers:
                target_key = f"task{i}_{h}"
                found = None
                
                # Búsqueda
                if target_key in raw_outputs: found = raw_outputs[target_key]
                if found is None:
                    for k in raw_outputs:
                        if target_key in k: found = raw_outputs[k]; break
                
                if found is not None:
                    arr = np.array(found)
                    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
                    flat_outputs.append(t)
                else:
                    flat_outputs.append(torch.zeros(1))
        return tuple(flat_outputs)

# --- FUNCIÓN PRINCIPAL ---
def verify_quantized(qhar_path, input_npy, golden_npz):
    print(f"🔌 Cargando Q-HAR: {qhar_path}")
    runner = ClientRunner(har=qhar_path)
    
    print("🚀 Levantando Emulador INT8...")
    hailo_net = PillarNest_Hailo_Module(runner)
    
    print("📂 Cargando datos...")
    input_data = np.load(input_npy)
    input_tensor = torch.from_numpy(input_data)
    golden_data = np.load(golden_npz)
    
    print("🏃 Ejecutando inferencia...")
    with torch.no_grad():
        outputs = hailo_net(input_tensor)

    print("\n⚖️  COMPARACIÓN (INT8 vs PyTorch Original):")
    print("-" * 90)
    print(f"{'CAPA':<20} | {'STATUS':<10} | {'ERROR (MAE)':<15} | {'MAX VAL'}")
    print("-" * 90)

    expected_names = []
    for i in range(6):
        for h in headers:
            expected_names.append(f"task{i}_{h}")

    total_error = 0
    for i, name in enumerate(expected_names):
        if i >= len(outputs): break
        hailo_tensor = outputs[i].numpy()
        
        if name in golden_data:
            gt_tensor = golden_data[name]
            diff = np.abs(gt_tensor - hailo_tensor).mean()
            max_val = np.max(np.abs(gt_tensor))
            total_error += diff
            
            # UMBRALES INT8
            # Heatmaps pueden tener algo de ruido (0.05).
            # Regresión (16-bit) debería ser muy precisa (< 0.01).
            limit = 0.02 if 'reg' in name or 'vel' in name else 0.08
            
            status = "✅ OK" if diff < limit else "⚠️ HIGH"
            print(f"{name:<20} | {status} | {diff:.8f}        | {max_val:.4f}")

    print("-" * 90)
    print(f"🏁 Error Global Promedio: {total_error/len(expected_names):.8f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('qhar', help='Ruta al .q.har generado')
    parser.add_argument('input', help='Ruta al input_tensor.npy')
    parser.add_argument('golden', help='Ruta al golden_outputs.npz')
    
    args = parser.parse_args()
    verify_quantized(args.qhar, args.input, args.golden)
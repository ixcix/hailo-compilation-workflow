import argparse
import numpy as np
import torch
import sys
import os
from hailo_sdk_client import ClientRunner, InferenceContext

# ==========================================
# 1. CLASE WRAPPER (Hailo -> PyTorch)
# ==========================================

# Headers
headers = ['reg', 'height', 'dim', 'rot', 'vel', 'iou', 'heatmap']

# # Prueba sin velocidad 
# headers = ['reg', 'height', 'dim', 'rot', 'iou', 'heatmap']

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
                raise ValueError("❌ El modelo Hailo no se cargó correctamente (None).")

    def forward(self, input_tensor):        
        # 1. Transponer a NHWC (PyTorch NCHW -> Hailo NHWC)
        input_nhwc = np.transpose(input_tensor.cpu().detach().numpy(), (0, 2, 3, 1))

        # 2. Inferencia en el Emulador
        try:
            raw_outputs = self._hailo_model({'input_pillars': input_nhwc})
        except:
            raw_outputs = self._hailo_model(input_nhwc)

        # =========================================================================
        # 🚑 RECUPERACIÓN DE NOMBRES DE SALIDA (BLINDADO)
        # =========================================================================
        if isinstance(raw_outputs, list):
            keys = None
            
            # --- INTENTO 1: Preguntar al Runner (La fuente de la verdad del HAR) ---
            try:
                # Esta función devuelve los nombres de los nodos finales del grafo
                keys = self._runner.get_hn_model().get_end_node_names()
                # A veces devuelve set, aseguramos lista ordenada si es posible, 
                # pero Keras suele respetar el orden del grafo.
                if keys:
                    keys = list(keys)
            except Exception as e:
                print(f"⚠️ Debug: Runner no devolvió nombres ({e})")

            # --- INTENTO 2: Atributos de Keras (Si el Runner falló) ---
            if not keys:
                keys = getattr(self._hailo_model, 'output_names', None)
            
            # --- INTENTO 3: Introspección Profunda (Solo si outputs existe) ---
            if (not keys) and hasattr(self._hailo_model, 'outputs'):
                if self._hailo_model.outputs is not None:
                    keys = []
                    for t in self._hailo_model.outputs:
                        # Limpieza: "task0_reg/Identity:0" -> "task0_reg"
                        if hasattr(t, 'name'):
                            clean_name = t.name.split(':')[0].split('/')[0]
                            keys.append(clean_name)
                        else:
                            keys.append(str(t))

            # --- ULTIMO RECURSO: Asignación Ciega (Esperemos que coincida) ---
            if keys is None or len(keys) != len(raw_outputs):
                print(f"⚠️  WARNING CRÍTICO: No hay nombres. Usando orden esperado a ciegas.")
                # Generamos los nombres en el orden que sabemos que exportamos
                # OJO: Esto asume que Keras respeta el orden de exportación de ONNX (suele hacerlo)
                keys = []
                for i in range(6):
                    for h in headers:
                        keys.append(f"task{i}_{h}")
                
                # Recortamos o rellenamos si no coincide la longitud
                if len(keys) != len(raw_outputs):
                    print(f"❌ ERROR: Esperábamos {len(keys)} salidas, recibimos {len(raw_outputs)}.")
                    # Fallback numérico para no romper el zip
                    keys = [f"out_{i}" for i in range(len(raw_outputs))]

            # Reconstruimos el diccionario
            # IMPORTANTE: Asumimos que la lista raw_outputs está alineada con keys
            raw_outputs = dict(zip(keys, raw_outputs))
        # =========================================================================

        flat_outputs = []
        
        
        # Recorremos 6 tareas x 7 cabeceras = 42 tensores
        for i in range(6):
            for h in headers:
                target_key = f"task{i}_{h}"
                found_tensor = None
                
                # Búsqueda 1: Exacta
                if target_key in raw_outputs:
                    found_tensor = raw_outputs[target_key]
                
                # Búsqueda 2: Aproximada (contiene el string)
                if found_tensor is None:
                    for key in raw_outputs.keys():
                        if target_key == key or key.startswith(f"{target_key}/"):
                            found_tensor = raw_outputs[key]
                            break
                
                if found_tensor is not None:
                    # Convertir Keras/Numpy -> Torch
                    if hasattr(found_tensor, 'numpy'):
                        arr = found_tensor.numpy()
                    else:
                        arr = np.array(found_tensor)
                        
                    out_tensor = torch.from_numpy(arr)
                    # Transponer de vuelta (Hailo NHWC -> PyTorch NCHW)
                    out_tensor = out_tensor.permute(0, 3, 1, 2).contiguous()
                    flat_outputs.append(out_tensor)
                else:
                    # Si no lo encontramos, imprimimos qué claves SI tenemos para debuggear
                    if i == 0 and h == 'reg': # Solo imprimir una vez
                        print(f"⚠️ Claves disponibles en emulador: {list(raw_outputs.keys())[:5]}...")
                    print(f"⚠️  CAPA PERDIDA: {target_key}")
                    flat_outputs.append(torch.zeros(1, 1, 1, 1)) 

        return tuple(flat_outputs)

# ==========================================
# 2. FUNCIÓN DE VERIFICACIÓN
# ==========================================
def verify_har(har_path, input_npy, golden_npz):
    print(f"🔌 Cargando HAR: {har_path}")
    
    # Cargar Runner
    runner = ClientRunner(har=har_path)
    
    # Hack para refrescar grafo FP si es necesario
    try:
        runner._sdk_backend.update_fp_model(runner._sdk_backend.model)
    except:
        pass

    # Instanciar nuestro Wrapper
    print("🚀 Iniciando Emulador (FP32)...")
    hailo_net = PillarNest_Hailo_Module(runner, emulate_quantized=False)
    
    # Cargar Datos
    print("📂 Cargando Golden Data...")
    try:
        input_data = np.load(input_npy)
        input_tensor = torch.from_numpy(input_data) 
        golden_data = np.load(golden_npz)
    except Exception as e:
        print(f"❌ Error leyendo archivos numpy: {e}")
        return
    
    # Ejecutar Inferencia
    print("🏃 Ejecutando forward pass...")
    with torch.no_grad():
        outputs = hailo_net(input_tensor) # Devuelve tupla de 42 tensores

    # Comparar
    print("\n⚖️  RESULTADOS DE LA COMPARACIÓN (Error Absoluto Medio):")
    print("-" * 80)
    print(f"{'CAPA':<20} | {'STATUS':<10} | {'ERROR (MAE)':<15} | {'MAX VAL'}")
    print("-" * 80)

    # Reconstruimos la lista de nombres esperados para iterar en orden
    expected_names = []
    for i in range(6):
        for h in headers:
            expected_names.append(f"task{i}_{h}")

    total_error = 0
    count = 0
    
    for i, name in enumerate(expected_names):
        if i >= len(outputs):
            break
        
        # Obtener tensor de Hailo (ya convertido a torch en el wrapper)
        hailo_tensor = outputs[i].numpy() 
        
        if name in golden_data:
            gt_tensor = golden_data[name]
            
            # Verificar shapes
            if hailo_tensor.shape != gt_tensor.shape:
                print(f"{name:<20} | ❌ SHAPE  | Got {hailo_tensor.shape} vs {gt_tensor.shape}")
                continue

            diff = np.abs(gt_tensor - hailo_tensor).mean()
            max_val = np.max(np.abs(gt_tensor))
            total_error += diff
            count += 1
            
            # Tolerancia FP32 (usualmente < 1e-4)
            status = "✅ OK" if diff < 1e-3 else "⚠️ HIGH"
            print(f"{name:<20} | {status} | {diff:.8f}        | {max_val:.4f}")
        else:
            print(f"{name:<20} | ❌ NO GT   | N/A")

    print("-" * 80)
    if count > 0:
        print(f"🏁 Error Global Promedio: {total_error/count:.8f}")
    else:
        print("⚠️ No se pudo comparar nada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('har', help='Ruta al archivo .har')
    parser.add_argument('input', help='Ruta al input_tensor.npy')
    parser.add_argument('golden', help='Ruta al golden_outputs.npz')
    
    args = parser.parse_args()
    verify_har(args.har, args.input, args.golden)
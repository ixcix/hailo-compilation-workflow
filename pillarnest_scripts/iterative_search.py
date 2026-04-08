import os
import tempfile
import shutil
import gc
import argparse
import numpy as np
import torch
import tensorflow as tf
from tqdm import tqdm
from hailo_sdk_client import ClientRunner, InferenceContext

# ==========================================
# 1. MOTOR MATEMÁTICO: LiDAR-Aware Asymmetric Grid Search (GPU)
# ==========================================
def iterative_search_ptq_lidar_gpu(fp32_tensor_np, num_bits=8):
    """
    Algoritmo Híbrido: Protege el límite inferior (negativos estructurales del LiDAR)
    y solo aplica Búsqueda Iterativa (MSE) sobre el límite superior (ruido/outliers).
    """
    flat_tensor = fp32_tensor_np.ravel()
    total_elements = flat_tensor.size
    chunk_size = 50_000_000
    
    # --- FASE 1: Buscar el mínimo y máximo globales reales ---
    x_min = float('inf')
    x_max = float('-inf')
    
    for i in range(0, total_elements, chunk_size):
        chunk_gpu = torch.tensor(flat_tensor[i:i+chunk_size], dtype=torch.float32, device='cuda')
        c_min = torch.min(chunk_gpu).item()
        c_max = torch.max(chunk_gpu).item()
        if c_min < x_min: x_min = c_min
        if c_max > x_max: x_max = c_max
        del chunk_gpu
        
    if x_min == x_max:
        return float(x_min), float(x_max)
        
    # --- FASE 2: Evaluar 100 candidatos (ENCOGIENDO SOLO EL TECHO) ---
    alphas = torch.linspace(0.7, 1.0, 100, device='cuda')
    sse_per_candidate = torch.zeros(100, dtype=torch.float32, device='cuda')
    
    for i in range(0, total_elements, chunk_size):
        chunk_gpu = torch.tensor(flat_tensor[i:i+chunk_size], dtype=torch.float32, device='cuda')
        
        for c_idx, alpha in enumerate(alphas):
            # 🎯 MAGIA AQUÍ: Límite inferior ANCLADO, límite superior ENCOGIDO
            c_min = x_min 
            c_max = x_max * alpha
            
            # Ecuación 15: Scale (s)
            s = (c_max - c_min) / ((2 ** num_bits) - 1)
            if s == 0: continue
            
            # Zero-Point (z)
            z = torch.round(-c_min / s)
            
            # Cuantización Asimétrica
            chunk_quant = torch.clamp(torch.round(chunk_gpu / s) + z, 0, (2 ** num_bits) - 1)
            chunk_dequant = (chunk_quant - z) * s
            
            # SSE
            sse_per_candidate[c_idx] += torch.sum((chunk_gpu - chunk_dequant) ** 2)
            
        del chunk_gpu

    # --- FASE 3: Encontrar el mejor techo ---
    best_idx = torch.argmin(sse_per_candidate).item()
    best_alpha = alphas[best_idx].item()
    
    best_min = x_min
    best_max = x_max * best_alpha
    
    torch.cuda.empty_cache()
    return float(best_min), float(best_max)

# ==========================================
# 2. EXTRACTOR HAILO (Vía Eager Monkey Patching)
# ==========================================
class PillarNest_Hailo_Extractor(torch.nn.Module):
    def __init__(self, runner):
        super().__init__()
        self._runner = runner
        self.hn = runner.get_hn_model()
        
        self.target_layers_full = [] 
        
        print("🔍 Analizando el grafo topológico para el PTQ...")
        
        for layer in self.hn.stable_toposort():
            layer_name = layer.name
            layer_type = type(layer).__name__

            if layer_type == 'FusedConv2DLayer':
                self.target_layers_full.append(layer_name)

        print(f"🎯 Se han interceptado {len(self.target_layers_full)} capas 'FusedConv2DLayer'.")
        
        print("⚙️  Levantando Emulador Nativo Keras (FP32)...")
        with self._runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            self._hailo_model = self._runner.get_keras_model(ctx)
            
            # --- EXTRACCIÓN SEGURA DE CAPAS REALES ---
            keras_layers_all = []
            def extract_real_layers(obj):
                if hasattr(obj, 'layers'):
                    layers_attr = obj.layers
                    if isinstance(layers_attr, dict):
                        layers_attr = layers_attr.values()
                    for l in layers_attr:
                        if isinstance(l, tf.keras.layers.Layer):
                            keras_layers_all.append(l)
                            extract_real_layers(l)
                            
            extract_real_layers(self._hailo_model)
            print(f"📦 Se han encontrado {len(keras_layers_all)} capas matemáticas reales en Keras.")

            # --- MONKEY PATCHING (El Hook de Keras) ---
            self.intermediate_outputs = {}
            self.mapped_layers = []
            
            def patch_layer(layer_obj, target_name):
                orig_call = layer_obj.call
                
                def new_call(*args, **kwargs):
                    out = orig_call(*args, **kwargs)
                    if hasattr(out, 'numpy'):
                        self.intermediate_outputs[target_name] = out.numpy()
                    return out
                    
                layer_obj.call = new_call
                self.mapped_layers.append(target_name)

            for layer_name in self.target_layers_full:
                nombre_corto = layer_name.split('/')[-1]
                
                for k_layer in keras_layers_all:
                    k_name = getattr(k_layer, 'name', '')
                    
                    if (k_name == layer_name or 
                        k_name == nombre_corto or 
                        k_name.endswith(f"_{nombre_corto}") or
                        f"{nombre_corto}_" in k_name):
                        
                        patch_layer(k_layer, layer_name)
                        break
            
            print(f"✅ Capas con Hook inyectado: {len(self.mapped_layers)} / {len(self.target_layers_full)}")
            
            if len(self.mapped_layers) == 0:
                print("❌ ERROR FATAL: Los nombres no coinciden.")
                exit(1)

    def forward(self, input_tensor):        
        input_np = input_tensor.cpu().detach().numpy()
        hn_shapes = self.hn.get_input_shapes()
        expected_shape = hn_shapes[0] 
        
        try:
            target_shape = (1, expected_shape[1], expected_shape[2], expected_shape[3])
            input_formatted = np.reshape(input_np, target_shape)
        except Exception as e:
            print(f"\n❌ Error de reshape: El modelo espera {target_shape} pero el numpy es {input_np.shape}.")
            raise e
        
        self.intermediate_outputs.clear()
        
        try:
            _ = self._hailo_model({'input_pillars': input_formatted})
        except:
            _ = self._hailo_model(input_formatted)

        return self.intermediate_outputs.copy()

# ==========================================
# 3. ORQUESTADOR: Inferencia Única + Disk Cache
# ==========================================
def run_iterative_search(har_path, calib_npy):
    runner = ClientRunner(har=har_path)
    extractor = PillarNest_Hailo_Extractor(runner)
    
    calib_data = np.load(calib_npy, mmap_mode='r')
    num_frames = calib_data.shape[0]
    
    # Procesaremos todos los frames del dataset que le pases (ej. 64)
    frames_a_procesar = num_frames
    print(f"\n🏃 Ejecutando Inferencia Única de {frames_a_procesar} frames...")

    # --- CONFIGURACIÓN DE TU DISCO 2 ---
    disco2_path = "/local/shared_with_docker/disco2_puente"
    os.makedirs(disco2_path, exist_ok=True)
    
    # Creamos la carpeta temporal DENTRO de tu disco 2
    temp_dir = tempfile.mkdtemp(prefix="hailo_ptq_cache_", dir=disco2_path)
    print(f"💾 Usando DISCO 2 como caché temporal masiva en: {temp_dir}")
    
    memmaps = {}

    # ==========================================
    # FASE 1: INFERENCIA Y VOLCADO A DISCO 2
    # ==========================================
    with torch.no_grad():
        for i in tqdm(range(frames_a_procesar), desc="Procesando Frames"):
            single_input = torch.from_numpy(calib_data[i:i+1].copy())
            outputs_dict = extractor(single_input)
            
            if i == 0:
                for k, v in outputs_dict.items():
                    shape = (frames_a_procesar,) + v.shape
                    safe_name = k.replace('/', '_') + '.dat'
                    filepath = os.path.join(temp_dir, safe_name)
                    memmaps[k] = np.memmap(filepath, dtype=v.dtype, mode='w+', shape=shape)
            
            for k, v in outputs_dict.items():
                memmaps[k][i] = v
                
            if i % 10 == 0:
                for k in memmaps:
                    memmaps[k].flush() 
                gc.collect()

    print("\n🔬 CALCULANDO MSE ÓPTIMO (LiDAR-Aware) EN GPU...")
    alls_commands = []
    
    # ==========================================
    # FASE 2: OPTIMIZACIÓN MATEMÁTICA EN GPU
    # ==========================================
    for layer_full in tqdm(extractor.target_layers_full, desc="Calculando Clipping (GPU)"):
        nombre_corto = layer_full.split('/')[-1]
        
        # Recuperamos el archivo de la caché
        safe_name = layer_full.replace('/', '_') + '.dat'
        filepath = os.path.join(temp_dir, safe_name)
        full_tensor_mmap = np.memmap(filepath, dtype='float32', mode='r')
        
        # Ejecutamos nuestra nueva función LiDAR-Aware
        best_min, best_max = iterative_search_ptq_lidar_gpu(full_tensor_mmap, num_bits=8)
        
        if best_min != 0.0 or best_max != 0.0:
            cmd = f"pre_quantization_optimization(activation_clipping, layers=[{nombre_corto}], mode=manual, clipping_values=[{best_min:.4f}, {best_max:.4f}])"
            alls_commands.append(cmd)
            
        del full_tensor_mmap
        gc.collect()

    # ==========================================
    # 4. SALIDA Y LIMPIEZA
    # ==========================================
    print(f"\n🧹 Limpiando caché del Disco 2 ({temp_dir})...")
    try:
        for k in list(memmaps.keys()):
            del memmaps[k]
        gc.collect()
        shutil.rmtree(temp_dir)
        print("✅ Disco 2 liberado correctamente.")
    except Exception as e:
        print(f"⚠️ Aviso: No se pudo borrar automáticamente. Borra {temp_dir} a mano. Error: {e}")

    print("\n" + "="*80)
    print("📋 COPIA ESTAS LÍNEAS EXACTAS EN TU ARCHIVO .ALLS:")
    print("="*80 + "\n")
    
    print("# === LiDAR-AWARE ITERATIVE SEARCH CLIPPING ===")
    for cmd in alls_commands:
        print(cmd)
        
    print("\n" + "="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('har', help='Ruta al archivo .har original')
    parser.add_argument('--calib', required=True, help='Dataset de calibración (.npy)')
    args = parser.parse_args()
    
    run_iterative_search(args.har, args.calib)
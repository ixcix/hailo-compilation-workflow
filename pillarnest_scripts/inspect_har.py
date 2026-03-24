from hailo_sdk_client import ClientRunner
import os

# Ajusta la ruta a tu HAR recién creado
har_path = "/local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_tiny_def/pillarnest_tiny_original_export.har" 

if not os.path.exists(har_path):
    print(f"❌ Error: No encuentro el archivo {har_path}")
    exit(1)

print(f"[INFO] Inspeccionando HAR: {har_path}")

# Cargar el HAR
runner = ClientRunner(har=har_path)

# Obtener el modelo interno (HN - Hailo Network)
hn = runner.get_hn_model()

print(f"\n{'LAYER NAME (Interno)':<30} | {'ORIGINAL NAME (Tuyo)':<30}")
print("-" * 65)

# Iterar sobre las capas de salida definidas en el HAR
# Nota: get_output_layers() devuelve objetos layer
output_layers = hn.get_output_layers()
print(f'Atributos de la clase: {dir(hn)}')
exit()
for layer in output_layers:
    # El nombre interno suele ser tipo "pillarnest_tiny/conv68"
    internal_name = layer.name
    # print(f"claves del layer: {dir(layer)}")  # Para ver qué atributos tiene el layer
    # exit()
    # El nombre original es el que viene del ONNX o del parseo
    # A veces se guarda en 'original_name' o en metadatos
    # original_name = getattr(layer, 'original_names', "N/A")
    original_name = layer.original_names 
    # Si no tiene atributo directo, a veces el nombre de la capa ES el original
    # si forzamos el 'end_node_names'.
    
    print(f"{internal_name:<30} | {original_name[0]:<30}")

print("-" * 65)
print(f"Total salidas detectadas: {len(output_layers)}")
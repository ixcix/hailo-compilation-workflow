from hailo_sdk_client import ClientRunner
import os

har_path = "/local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_tiny_def/pillarnest_tiny_original_export.har" 
txt_output_path = "mapa_completo_red.txt"

if not os.path.exists(har_path):
    print(f"❌ Error: No encuentro el archivo {har_path}")
    exit(1)

print(f"[INFO] Inspeccionando HAR: {har_path}")

runner = ClientRunner(har=har_path)
hn = runner.get_hn_model()

# Capa donde termina el backbone
ULTIMA_CAPA_BACKBONE = "conv22" 

backbone_layers_to_freeze = []
en_backbone = True
encontrado_limite = False

print(f"💾 Guardando inspección completa en: {txt_output_path} ...\n")

with open(txt_output_path, "w", encoding="utf-8") as f:
    # Escribimos la cabecera en el archivo
    cabecera = f"{'IDX':<5} | {'NOMBRE INTERNO (Hailo)':<60} | {'TIPO':<15} | {'ACCIÓN ASIGNADA':<25} | {'NOMBRE ORIGINAL'}"
    f.write(cabecera + "\n")
    f.write("-" * 140 + "\n")

    # Recorremos TODAS las capas de la red de principio a fin
    for idx, layer in enumerate(hn.stable_toposort()):
        
        # Extraemos atributos
        layer_name = layer.name
        layer_type = type(layer).__name__
        original_names = getattr(layer, 'original_names', [])
        orig_name = original_names[0] if original_names else "N/A"

        # ¿Tiene parámetros que se puedan entrenar/congelar?
        es_entrenable = False
        if "Conv" in layer_type or "Norm" in layer_type or "Batch" in layer_type:
            es_entrenable = True

        # Asignamos la acción dependiendo de si estamos en el Backbone o en las Cabezas
        if en_backbone:
            if es_entrenable:
                accion = "❄️ CONGELAR (Backbone)"
                # --- MODIFICACIÓN AQUÍ ---
                # Cortamos por la barra '/' y nos quedamos con el último elemento
                nombre_corto = layer_name.split('/')[-1]
                backbone_layers_to_freeze.append(nombre_corto)
            else:
                accion = "⏭️ SALTAR (Sin pesos)"
        else:
            if es_entrenable:
                accion = "🔥 ENTRENAR (Head)"
            else:
                accion = "⏭️ SALTAR (Sin pesos)"
        
        # 1. Escribimos la línea en el archivo de texto
        linea = f"{idx:<5} | {layer_name:<60} | {layer_type:<15} | {accion:<25} | {orig_name}\n"
        f.write(linea)
        
        # 2. Imprimimos por consola de forma resumida
        print(f"{layer_name:<50} | {accion}")

        # Comprobamos si hemos alcanzado la capa límite para bajar el interruptor
        if ULTIMA_CAPA_BACKBONE in layer_name and en_backbone:
            encontrado_limite = True
            en_backbone = False  # A partir de aquí, dejamos de congelar

print("\n" + "=" * 80)
if encontrado_limite:
    print(f"✅ Límite del backbone encontrado correctamente en la capa que contiene: '{ULTIMA_CAPA_BACKBONE}'")
else:
    print(f"⚠️ AVISO: No se encontró la capa '{ULTIMA_CAPA_BACKBONE}'. Se ha congelado toda la red.")

print(f"✅ Total de capas entrenables congeladas en el backbone: {len(backbone_layers_to_freeze)}")
print(f"📂 El mapa detallado de TODAS las capas se ha guardado en: {txt_output_path}")
print("=" * 80)

# Imprimir la lista formateada lista para pegar en el .alls
print("\n📋 CÓPIA ESTA LÍNEA EXACTA EN TU ARCHIVO .ALLS:")
print("layers_to_freeze=[", end="")
print(", ".join(backbone_layers_to_freeze), end="")
print("]\n")
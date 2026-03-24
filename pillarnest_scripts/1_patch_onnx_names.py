import argparse
import onnx
import sys

def rename_output_nodes(onnx_path, output_path=None):
    print(f"🔧 Abriendo modelo: {onnx_path}")
    
    # Cargar el modelo
    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        print(f"❌ Error cargando ONNX: {e}")
        sys.exit(1)

    graph = model.graph
    
    # 1. Identificar cuáles son los tensores de salida finales del grafo
    # Estos son los nombres bonitos que pusiste en PyTorch (ej: 'task0_heatmap')
    graph_output_names = set([out.name for out in graph.output])
    
    print(f"📋 Salidas detectadas en el grafo: {list(graph_output_names)}")
    
    nodes_renamed = 0
    
    # 2. Recorrer todos los nodos para encontrar quién produce esas salidas
    for node in graph.node:
        # Verificamos si alguna de las salidas de este nodo es una salida del grafo
        for node_output in node.output:
            if node_output in graph_output_names:
                
                old_name = node.name
                
                # Naming Convention: 
                # Si el tensor se llama "task0_heatmap", llamamos a la capa "task0_heatmap_Layer"
                # Esto ayuda a distinguir en el Hailo Profiler qué es dato y qué es computación.
                new_name = f"{node_output}"
                
                # Asignar el nuevo nombre al nodo
                node.name = new_name
                
                print(f"   ✨ Renombrando Nodo: '{old_name}' -> '{new_name}' (Output: {node_output})")
                nodes_renamed += 1
                
                # Nota: Un nodo puede tener múltiples salidas, pero en las cabeceras de detección
                # suele ser 1 a 1. Si hubiera conflicto, este script renombra basado en la última coincidencia,
                # lo cual suele ser correcto para capas finales.

    if nodes_renamed == 0:
        print("⚠️  ADVERTENCIA: No se encontraron nodos conectados directamente a las salidas. "
              "¿Seguro que el ONNX tiene los output_names correctos?")
    else:
        print(f"✅ Se renombraron {nodes_renamed} nodos exitosamente.")

    # 3. Guardar
    save_path = output_path if output_path else onnx_path.replace(".onnx", "_named.onnx")
    onnx.save(model, save_path)
    print(f"💾 Modelo guardado en: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renombra los nodos finales de un ONNX para coincidir con sus tensores de salida.")
    parser.add_argument("input_onnx", help="Ruta al archivo ONNX original")
    parser.add_argument("--output", help="Ruta de salida (opcional)", default=None)
    
    args = parser.parse_args()
    
    rename_output_nodes(args.input_onnx, args.output)
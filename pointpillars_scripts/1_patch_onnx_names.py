import argparse
import onnx
import sys

def patch_pointpillars_outputs(onnx_path, output_path=None):
    print(f"🧐 Analizando ONNX: {onnx_path}")
    
    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        sys.exit(1)

    graph = model.graph
    
    # 1. Ver qué nombres de salida tiene el grafo actualmente
    # Esto es lo que definiste en el export de PyTorch
    graph_output_names = [out.name for out in graph.output]
    print(f"🔍 Salidas encontradas en el ONNX: {graph_output_names}")

    if len(graph_output_names) != 3:
        print(f"⚠️  CUIDADO: Se esperaban 3 salidas para PointPillars, pero he encontrado {len(graph_output_names)}.")
        print("   Asegúrate de que no estás exportando múltiples niveles de FPN innecesarios.")

    nodes_renamed = 0
    
    # 2. Renombrar los NODOS (capas) para que coincidan con sus TENSORES de salida
    # Hailo utiliza el nombre del NODO para generar el nombre en el HEF final.
    for node in graph.node:
        for node_output in node.output:
            if node_output in graph_output_names:
                old_node_name = node.name
                
                # Forzamos que el nombre del nodo sea idéntico al del tensor de salida
                new_node_name = node_output
                node.name = new_node_name
                
                print(f"   ✨ Nodo parcheado: '{old_node_name}' -> '{new_node_name}'")
                nodes_renamed += 1

    if nodes_renamed == 0:
        print("❌ ERROR: No se ha podido renombrar ningún nodo. Los nombres de salida del ONNX no coinciden con las capas finales.")
        return

    # 3. Guardar el modelo parcheado
    final_path = output_path if output_path else onnx_path.replace(".onnx", "_fixed.onnx")
    onnx.save(model, final_path)
    print(f"\n✅ Proceso completado. {nodes_renamed} nodos fijados.")
    print(f"💾 Nuevo ONNX listo para Hailo en: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fija los nombres de los nodos de salida para el compilador de Hailo.")
    parser.add_argument("input", help="ONNX de PointPillars simplificado")
    parser.add_argument("--output", help="Ruta del ONNX de salida", default=None)
    
    args = parser.parse_args()
    patch_pointpillars_outputs(args.input, args.output)
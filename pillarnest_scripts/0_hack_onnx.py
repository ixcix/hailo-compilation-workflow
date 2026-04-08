import onnx
from onnx import helper

def hack_onnx_for_hailo(input_path, output_path):
    print(f"🕵️  Cargando modelo para cirugía: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph

    nodes_modified = 0

    for node in graph.node:
        if node.op_type == "ConvTranspose":
            print(f"  🔧 Procesando nodo: {node.name}")
            
            # 1. Filtrar atributos prohibidos (pads y output_padding)
            # Si estos atributos existen, el compilador de Hailo ignora el auto_pad.
            new_attributes = [
                attr for attr in node.attribute 
                if attr.name not in ['pads', 'output_padding', 'auto_pad']
            ]
            
            # 2. Limpiar atributos actuales
            node.ClearField('attribute')
            
            # 3. Reinyectar atributos limpios + el HACK de SAME_UPPER
            node.attribute.extend(new_attributes)
            
            # Usamos SAME_UPPER que es el estándar ONNX para lo que Hailo llama SAME_TENSORFLOW
            auto_pad_attr = helper.make_attribute("auto_pad", "SAME_TENSORFLOW")
            node.attribute.append(auto_pad_attr)
            
            nodes_modified += 1

    # 4. Guardar y verificar
    onnx.save(model, output_path)
    print(f"✅ Cirugía completada. {nodes_modified} nodos modificados.")
    print(f"📦 Modelo listo para Hailo: {output_path}")

if __name__ == "__main__":
    # Cambia estos paths a los tuyos
    hack_onnx_for_hailo(
        "model/hailo8l/pillarnest_small/pillarnest_small_hailo_v2.onnx", 
        "model/hailo8l/pillarnest_small/pillarnest_small_hailo_v2_hacked.onnx"
    )
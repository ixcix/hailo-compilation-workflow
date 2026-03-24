import onnx
# No HACE FALTAAAAAAAAAAAAAAAAAAA

# Cargar el modelo original
working_dir = "model/hailo8l/pillarnest_tiny_def"
model_path = f"{working_dir}/pillarnest_tiny_original_export.onnx"
hacked_path = f"{working_dir}/pillarnest_hailo_fixed.onnx"

print(f"Cargando modelo: {model_path}")
model = onnx.load(model_path)

modifications = 0
for node in model.graph.node:
    # Buscar solo los nodos ConvTranspose
    if node.op_type == "ConvTranspose":
        # Guardar todos los atributos que NO sean 'pads'
        new_attrs = [a for a in node.attribute if a.name != 'pads']
        
        # Limpiar los atributos del nodo
        del node.attribute[:]
        
        # Restaurar los atributos limpios
        node.attribute.extend(new_attrs)
        
        # ¡LA MAGIA! Inyectar SAME_UPPER (que Hailo lee como SAME_TENSORFLOW)
        auto_pad_attr = onnx.helper.make_attribute("auto_pad", b"SAME_UPPER")
        node.attribute.append(auto_pad_attr)
        
        modifications += 1

# Guardar el nuevo modelo
onnx.save(model, hacked_path)
print(f"Éxito: Se modificaron {modifications} nodos ConvTranspose.")
print(f"Modelo guardado como: {hacked_path}")
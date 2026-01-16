import onnx
from onnxsim import simplify


onnx_name = "/local/shared_with_docker/hailo-compilation-workflow/model/pillarnest_tiny_hailo.onnx"
onnx_name_simp = "/local/shared_with_docker/hailo-compilation-workflow/model/pillarnest_tiny_hailo_simplified.onnx"

model = onnx.load(onnx_name)

model_simplified, check = simplify(model)

if check:
    onnx.save(model_simplified, f"{onnx_name_simp}")
    print(f"Modelo simplificado guardado en {onnx_name_simp}")
else:
    print("La simplificación del modelo falló.")

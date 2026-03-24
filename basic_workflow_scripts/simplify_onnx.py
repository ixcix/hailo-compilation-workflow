import onnx
from onnxsim import simplify

name = ""
onnx_name = f"../model/{name}.onnx"
onnx_name_simp = f"../model/{name}_simp.onnx"

model = onnx.load(onnx_name)

model_simplified, check = simplify(model)

if check:
    onnx.save(model_simplified, f"{onnx_name_simp}")
    print(f"Modelo simplificado guardado en {onnx_name_simp}")
else:
    print("La simplificación del modelo falló.")

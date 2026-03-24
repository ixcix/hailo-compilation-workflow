import onnx
from onnxsim import simplify
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Renombra los nodos finales de un ONNX para coincidir con sus tensores de salida.")
    parser.add_argument("input_onnx", help="Ruta al archivo ONNX original")
    parser.add_argument("--output", help="Ruta de salida (opcional)", default=None)

    args = parser.parse_args()
        


    onnx_path = args.input_onnx
    onnx_simp_path = args.output if args.output else onnx_path.replace(".onnx", "_simp.onnx")

    model = onnx.load(onnx_path)
    model_simplified, check = simplify(model)

    if check:
        onnx.save(model_simplified, f"{onnx_simp_path}")
        print(f"Modelo simplificado guardado en {onnx_simp_path}")
    else:
        print("La simplificación del modelo falló.")

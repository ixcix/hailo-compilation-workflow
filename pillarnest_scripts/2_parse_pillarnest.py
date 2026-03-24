import os
import argparse

# --- 1. Configuración de Entorno y Logs  ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)


# Configurar variables de entorno para redirigir los logs de Hailo SDK y HailoRT
os.environ['HAILO_SDK_LOG_DIR'] = logs_dir
os.environ['HAILORT_LOGGER_PATH'] = logs_dir


from hailo_sdk_client import ClientRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renombra los nodos finales de un ONNX para coincidir con sus tensores de salida.")
    parser.add_argument("input_onnx", help="Ruta al archivo ONNX original")
    parser.add_argument("--output", help="Path del .har (modeo parseado)", default=None)
    
    args = parser.parse_args()


    # 1. Configuración
    onnx_path = args.input_onnx
    har_name = os.path.basename(onnx_path).replace("_simp_named.onnx", ".har")
    har_path = args.output if args.output else onnx_path.replace("_simp_named.onnx", ".har")
    chosen_hw_arch = "hailo8l" 

    # 2. Generar la lista de 42 nombres EXACTOS
    # El orden aquí es crucial. Debe coincidir con el orden de salida del ONNX.
    # Basado en tu exportador: 6 tareas x 7 cabezales.
    output_names = []
    heads = ['reg', 'height', 'dim', 'rot', 'vel', 'iou', 'heatmap']

    #prueba sin velocidad
    # heads = ['reg', 'height', 'dim', 'rot', 'iou', 'heatmap']

    for i in range(6): # 5 Tareas sin veloc
        for head in heads:
            output_names.append(f"task{i}_{head}")

    print(f"[INFO] Nombres de salida ({len(output_names)}):")
    print(f"      {output_names[:6]} ...")

    # 3. Inicializar Runner
    runner = ClientRunner(hw_arch=chosen_hw_arch)

    print(f"[INFO] Parseando modelo: {onnx_path}")

    runner.translate_onnx_model(
        onnx_path,
        har_name,
        start_node_names=["input_canvas"], 
        end_node_names=output_names
    )

    # 5. Guardar HAR
    runner.save_har(har_path)
    print(f"[INFO] HAR guardado en: {har_path}")
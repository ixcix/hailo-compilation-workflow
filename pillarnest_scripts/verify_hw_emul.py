import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType

def verify_hw_with_hef_truth(hef_path, input_npy, golden_npz):
    hef = HEF(hef_path)
    golden_data = np.load(golden_npz)
    input_data = np.load(input_npy).astype(np.float32)

    # 1. Extraemos la info de cuantización de los outputs directamente del HEF
    output_info_map = {}
    for info in hef.get_output_vstream_infos():
        output_info_map[info.name] = {
            'original_names': hef.get_original_names_from_vstream_name(info.name),
            'scale': info.quant_info.qp_scale,
            'zp': info.quant_info.qp_zp,
            'shape': info.shape
        }

    # 2. Configurar Hardware
    target = VDevice()
    network_group = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    
    # IMPORTANTE: Pedimos FLOAT32 para que la HailoRT aplique el scale/zp por nosotros
    # Si esto falla, lo haremos a mano con los datos UINT8
    v_in_p = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
    v_out_p = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

    # 3. Inferencia
    print("🚀 Ejecutando inferencia física...")
    with InferVStreams(network_group, v_in_p, v_out_p) as infer_pipeline:
        with network_group.activate(network_group.create_params()):
            # OJO: Asegúrate de que input_data sea NHWC si el HEF lo pide
            raw_results = infer_pipeline.infer({hef.get_input_vstream_infos()[0].name: input_data})

    # 4. Mapeo y Comparación
    print("\n⚖️ COMPARACIÓN USANDO NOMBRES ORIGINALES DEL HEF:")
    for vstream_name, data in raw_results.items():
        info = output_info_map[vstream_name]
        
        # Si el vStream agrupa varias capas originales (como suele pasar en CenterPoint)
        # tenemos que tener cuidado, pero normalmente es 1 a 1 o se busca por nombre
        for orig_name in info['original_names']:
            if orig_name in golden_data:
                gt = golden_data[orig_name]
                hw = data
                
                # Transponer si el HW devuelve NHWC y el Golden es NCHW
                if hw.ndim == 4 and hw.shape[3] == gt.shape[1]:
                    hw = hw.transpose(0, 3, 1, 2)
                
                hw = hw.reshape(gt.shape)
                mae = np.abs(hw - gt).mean()
                print(f"{'✅' if mae < 0.1 else '⚠️'} {orig_name:<20} | MAE: {mae:.6f} | VStream: {vstream_name}")
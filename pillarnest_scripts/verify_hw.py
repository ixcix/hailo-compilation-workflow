import argparse
import numpy as np
import sys
import os

from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType)

headers = ['reg', 'height', 'dim', 'rot', 'vel', 'iou', 'heatmap']

def run_inference_cycle(network_group, input_vstream_name, input_data, golden_data, output_nodes_info, v_in_p, v_out_p, mode_name):
    print(f"\n\n{'='*30} MODO: {mode_name} {'='*30}")
    
    # Asegurar tipo y contigüidad
    input_nhwc = np.ascontiguousarray(input_data).astype(np.float32)

    print(f"🕵️‍♂️ SONDA 1: INPUT AL CHIP")
    print(f"  Shape enviado: {input_nhwc.shape}")
    print(f"  Rango: [{input_nhwc.min():.4f}, {input_nhwc.max():.4f}] | Media: {input_nhwc.mean():.4f}")

    with InferVStreams(network_group, v_in_p, v_out_p) as infer_pipeline:
        # Inferencia
        infer_results = infer_pipeline.infer({input_vstream_name: input_nhwc})

    output_map = {}
    
    # Parseo de resultados
    for stream_name, data in infer_results.items():
        if data.ndim == 4: # NHWC -> NCHW
            data_nchw = np.ascontiguousarray(data.transpose(0, 3, 1, 2))
        elif data.ndim == 3: # HWC -> NCHW
            data_nchw = np.ascontiguousarray(data.transpose(2, 0, 1)[None, :, :, :])
        else:
            data_nchw = data

        for orig_name in output_nodes_info[stream_name]['original_names']:
            output_map[orig_name] = data_nchw

    # Comparación
    print("\n⚖️ COMPARACIÓN:")
    print(f"{'CAPA':<20} | {'STATUS':<10} | {'MAE':<15}")
    print("-" * 50)

    total_error = 0
    checked_layers = 0
    
    for i in range(6):
        for h in headers:
            name = f"task{i}_{h}"
            if name in golden_data and name in output_map:
                gt = golden_data[name]
                hw = output_map[name]
                if hw.shape != gt.shape: hw = hw.reshape(gt.shape)
                
                diff = np.abs(gt - hw).mean()
                total_error += diff
                checked_layers += 1
                status = "✅ OK" if diff < 0.1 else "⚠️ HIGH"
                print(f"{name:<20} | {status} | {diff:.8f}")

    if checked_layers > 0:
        print(f"\n🏁 ERROR GLOBAL [{mode_name}]: {total_error/checked_layers:.8f}")

def verify_hardware_dual(hef_path, input_npy, golden_npz):
    target = VDevice()
    hef = HEF(hef_path)
    network_group = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    
    # Metadatos
    output_nodes_info = {info.name: {"original_names": hef.get_original_names_from_vstream_name(info.name)} 
                         for info in hef.get_output_vstream_infos()}
    input_vstream_info = hef.get_input_vstream_infos()[0]
    input_vstream_name = input_vstream_info.name
    
    v_in_p = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
    v_out_p = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

    print("📂 Cargando datos...")
    
    raw_input = np.load(input_npy)
    golden_data = np.load(golden_npz)

    with network_group.activate(network_group.create_params()):
        # PASADA 1: TAL CUAL VIENE EL .NPY
        run_inference_cycle(network_group, input_vstream_name, raw_input, golden_data, 
                            output_nodes_info, v_in_p, v_out_p, "ORIGINAL (AS IS)")

        # PASADA 2: FORZANDO TRANSPOSE (SI ES NCHW)
        if raw_input.shape[1] == 48:
            input_transposed = np.transpose(raw_input, (0, 2, 3, 1))
            run_inference_cycle(network_group, input_vstream_name, input_transposed, golden_data, 
                                output_nodes_info, v_in_p, v_out_p, "FORCED TRANSPOSE (NHWC)")
        else:
            print("\n⚠️ El input no parece NCHW, saltando Pasada 2.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('hef')
    parser.add_argument('input')
    parser.add_argument('golden')
    args = parser.parse_args()
    verify_hardware_dual(args.hef, args.input, args.golden)
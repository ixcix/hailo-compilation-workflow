import os
import argparse
import json
# un cagarro solo sirve para modelos pequeños 
# 1. FIX DE LA GRÁFICA: Forzar modo headless antes de importar pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from hailo_sdk_client import ClientRunner

def analyze_centerpoint_snr(runner):
    print("\n📊 Extrayendo estadísticas de ruido de Hailo...")
    params_statistics = runner.get_params_statistics()
    
    output_layers = [layer.name for layer in runner.get_hn_model().get_output_layers()]
    output_snr_results = {}

    for out_layer in output_layers:
        snr_key = f"{out_layer}/layer_noise_analysis/noise_results/{out_layer}"
        layer_snr = params_statistics.get(snr_key)
        if layer_snr is not None:
            output_snr_results[out_layer] = layer_snr[0].tolist()
        else:
            output_snr_results[out_layer] = 0.0 

    heatmaps_snr = {k: v for k, v in output_snr_results.items() if "heatmap" in k.lower()}
    regression_snr = {k: v for k, v in output_snr_results.items() if "heatmap" not in k.lower()}

    print("\n🔥 PEORES 3 HEATMAPS (SNR más bajo):")
    worst_heatmaps = sorted(heatmaps_snr.items(), key=lambda x: x[1])[:3]
    for name, snr in worst_heatmaps: print(f"  - {name}: {snr:.2f} dB")

    print("\n📐 PEORES 3 REGRESIONES (SNR más bajo):")
    worst_regs = sorted(regression_snr.items(), key=lambda x: x[1])[:3]
    for name, snr in worst_regs: print(f"  - {name}: {snr:.2f} dB")

    return output_snr_results

def plot_centerpoint_snr(snr_dict):
    sorted_snr = sorted(snr_dict.items(), key=lambda x: x[1])
    layers = [x[0].split('/')[-1] for x in sorted_snr] 
    snrs = [x[1] for x in sorted_snr]

    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(layers, snrs, color=['red' if s < 10 else 'green' for s in snrs])
    
    ax.axhline(y=10, color='orange', linestyle='--', label='Umbral Aceptable (10 dB)')
    
    plt.title("Signal-to-Noise Ratio (SNR) por Capa de Salida (INT8 vs FP32)", fontsize=14)
    plt.xlabel("Capas de Salida", fontsize=12)
    plt.ylabel("SNR (dB)", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.legend()
    plt.tight_layout()
    
    output_img = "snr_analysis_centerpoint.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n📈 Gráfica guardada exitosamente como '{output_img}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--har", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--count", type=int, default=10)
    # 2. FIX DE LA VRAM: Añadimos argumento para forzar CPU
    parser.add_argument("--cpu", action="store_true", help="Fuerza la ejecución en CPU para evitar OOM de la GPU")
    args = parser.parse_args()

    if args.cpu:
        print("🖥️  Forzando ejecución en CPU (CUDA_VISIBLE_DEVICES = -1)...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print(f"🚀 Iniciando análisis de ruido...")
    runner = ClientRunner(har=args.har)

    print("\n⏳ Ejecutando Analyze Noise...")
    runner.analyze_noise(args.data, batch_size=1, data_count=args.count)

    # 3. FIX DEL HAR: ¡Eliminado runner.save_har()!

    snr_results = analyze_centerpoint_snr(runner)
    plot_centerpoint_snr(snr_results)

    # 4. FIX PARA EL PAPER: Guardamos los datos brutos por si los necesitas para una tabla en LaTeX
    with open("snr_results.json", "w") as f:
        json.dump(snr_results, f, indent=4)
    print("💾 Datos brutos guardados en 'snr_results.json'")

if __name__ == "__main__":
    main()
import numpy as np
import argparse
import matplotlib.pyplot as plt

def compare_datasets(calib_path, golden_path):
    print(f"📊 Cargando Calib Dataset: {calib_path}")
    calib_data = np.load(calib_path)
    
    print(f"📊 Cargando Golden Data: {golden_path}")
    golden_data = np.load(golden_path)

    print("-" * 60)
    print(f"{'STAT':<15} | {'CALIB DATA':<20} | {'GOLDEN DATA':<20}")
    print("-" * 60)
    
    # Shape
    print(f"{'Shape':<15} | {str(calib_data.shape):<20} | {str(golden_data.shape):<20}")
    
    # Min/Max Global
    print(f"{'Min Val':<15} | {calib_data.min():<20.4f} | {golden_data.min():<20.4f}")
    print(f"{'Max Val':<15} | {calib_data.max():<20.4f} | {golden_data.max():<20.4f}")
    print(f"{'Mean':<15}    | {calib_data.mean():<20.4f} | {golden_data.mean():<20.4f}")
    print(f"{'Std Dev':<15} | {calib_data.std():<20.4f} | {golden_data.std():<20.4f}")
    
    # Análisis de "Sparsity" (Cuántos ceros hay)
    # Esto es vital en LiDAR. Si uno tiene muchos más ceros que el otro, la distribución cambia.
    zeros_calib = np.mean(calib_data == 0) * 100
    zeros_golden = np.mean(golden_data == 0) * 100
    print(f"{'% Zeros':<15} | {zeros_calib:<20.2f}% | {zeros_golden:<20.2f}%")
    
    print("-" * 60)

    # Diagnóstico
    max_diff = abs(calib_data.max() - golden_data.max())
    if max_diff > 10.0:
        print("❌ ALERTA ROJA: Los rangos máximos son MUY diferentes.")
        print("   El cuantizador ajustó la escala para el valor más grande, perdiendo precisión en los pequeños.")
    elif abs(zeros_calib - zeros_golden) > 10.0:
        print("⚠️ ALERTA: La densidad de puntos es muy diferente.")
    else:
        print("✅ PARECE CORRECTO: Las estadísticas son similares.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("calib", help="Ruta al npy de calibración")
    parser.add_argument("golden", help="Ruta al npy golden")
    args = parser.parse_args()
    
    compare_datasets(args.calib, args.golden)
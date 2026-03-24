import numpy as np
import matplotlib.pyplot as plt
import argparse

def inspect_calib(npy_path):
    print(f"🕵️ Cargando: {npy_path}")
    data = np.load(npy_path) # (64, 720, 720, 48)
    
    # Vamos a inspeccionar el frame 0 y el frame 32
    for idx in [0, 32]:
        if idx >= len(data): break
        
        frame = data[idx] # (720, 720, 48)
        
        # Aplanamos canales para ver "dónde hay cosas" (max projection)
        # Si esto sale negro, el dataset está vacío.
        bev_map = np.max(frame, axis=-1)
        
        print(f"\n--- FRAME {idx} ---")
        print(f"Max Value: {np.max(frame):.4f}")
        print(f"Min Value: {np.min(frame):.4f}")
        
        # Histograma de valores NO CERO
        non_zeros = frame[frame > 0.1]
        
        plt.figure(figsize=(15, 5))
        
        # 1. Mapa visual
        plt.subplot(1, 2, 1)
        plt.imshow(bev_map, cmap='gray', vmin=0, vmax=5) # Saturamos a 5 para ver algo
        plt.title(f"Frame {idx} - BEV Projection")
        
        # 2. Histograma Logarítmico
        plt.subplot(1, 2, 2)
        plt.hist(non_zeros.flatten(), bins=100, log=True, color='blue')
        plt.title(f"Distribución de valores > 0.1")
        plt.xlabel("Valor del Voxel")
        plt.ylabel("Frecuencia (Log)")
        
        plt.savefig(f"debug_calib_frame_{idx}.png")
        print(f"📸 Guardado debug_calib_frame_{idx}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy")
    args = parser.parse_args()
    inspect_calib(args.npy)
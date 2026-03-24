from hailo_platform import HEF
import argparse

def inspect_hef(hef_path):
    print(f"🔍 Inspeccionando archivo HEF: {hef_path}")
    
    # Cargar HEF
    try:
        hef = HEF(hef_path)
    except Exception as e:
        print(f"❌ Error cargando HEF: {e}")
        return

    # Obtener info de streams
    input_vstream_infos = hef.get_input_vstream_infos()
    output_vstream_infos = hef.get_output_vstream_infos()
    print(f"metodos hef: {dir(hef)}")  # Para ver qué métodos tiene el HEF

    print("\n" + "="*60)
    print("📥 INPUTS (Lo que debes configurar en tu Scatter)")
    print("="*60)
    
    for i, info in enumerate(input_vstream_infos):
        print(f"Input #{i}: {info.name}")
        print(f"   Shape: {info.shape}")
        print(f"   Format: {info.format.type}")
        
        # AQUÍ ESTÁ LA MAGIA
        scale = info.quant_info.qp_scale
        zp = info.quant_info.qp_zp
        
        print(f"   👉 Scale:      {scale:.8f}")
        print(f"   👉 Zero Point: {zp}")
        print("-" * 30)

    print("\n" + "="*60)
    print("📤 OUTPUTS (Lo que recibirás del Hailo)")
    print("="*60)
    for i, info in enumerate(output_vstream_infos):
        original_names = hef.get_original_names_from_vstream_name(info.name)
        print(f"Output #{i}: {info.name}")
        print(f"Original layers: {original_names}")
        print(f"   Shape: {info.shape}")
        scale = info.quant_info.qp_scale
        zp = info.quant_info.qp_zp
        print(f"   👉 Scale:      {scale:.8f}")
        print(f"   👉 Zero Point: {zp}")
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hef", help="Ruta al archivo .hef")
    args = parser.parse_args()
    inspect_hef(args.hef)
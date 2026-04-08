
# Hailo compilation workflow

Repositorio que permite seguir todos los pasos para pasar de un modelo (tflite, onnx etc) al formajo ejecutable en el acelerador Hailo

## SOFTWARE DE HAILO 

El SW de hailo empleado: AI Software Suite: Docker 2025-01

El SW de hailo estrictamente necesario es el 
- DFC (Data flow compiler): compilador de hailo 
- HailoRT: API en tiempo de ejecución para la comunicación entre el software y el hardware de Hailo.
También se puede descargar el AI Software Suite que contiene además el Model Zoo y TAPPAS (ver documentación)

Ojo con las versiones, compatibilidad de versiones por si se quiere descargar por separado 
```text
| AI SW Suite   | Dataflow Compiler | HailoRT   | Integration Tool  | Model Zoo | TAPPAS    |
| 2025-01       | v3.30.0           | v4.20.0   | v1.20.0           | v2.14.0   | v3.31.0   |

```

EN caso de usar el DOCKER DE HAILO (una vez lo hayas creado)

``` bash
    cd ~/path/docker_oficial_hailo
    sudo rm -r /tmp/hailo_docker.xauth
    ./hailo_ai_sw_suite_docker_run.sh --resume
```

## OPTIMIZACION AVANZADA

### Aumento de swap 
Cuando tienes un error OOM es pq no hay suficiente RAM, para ello asignamos espacio para aumentar la swap usando el disco 1 (64 GB)
```bash
# 1. Crear el archivo (La / inicial asegura que sea en el Disco 1, fuera de carpetas de usuario)
sudo fallocate -l 64G /hailo_swapfile

# 2. Darle permisos (Solo el sistema puede tocarlo, esto es por seguridad)
sudo chmod 600 /hailo_swapfile

# 3. Formatearlo como RAM
sudo mkswap /hailo_swapfile

# 4. Activar la RAM Virtual
sudo swapon /hailo_swapfile

# Ver que la swap ha aumentado:
free -h

# IMPORTANTE deshacer esto para no tener 64gb del disco ocupados con swap:
sudo swapoff /hailo_swapfile

sudo rm /hailo_swapfile

```

### Problemas VRAM 

# 1. Forzamos la reserva total de VRAM
export HAILO_SET_MEMORY_GROWTH=false

### Aumento de memoria para tmp files de algoritmos pesados (niveles de opt altos)
Si quiero usar el segundo disco para los tmp files (adaround, finetuning... OCUPAN MUCHISIMO). Hago este apaño. 
El SDK de Hailo y sus dependencias (TensorFlow) generan una cantidad masiva de datos temporales. Por defecto, estos datos se escriben en el Directorio de Trabajo Actual (CWD)
Por ello es necesario:
- Crear una carpeta puente dentro del shared docker a un disco con espacio suficiente (mas de 300GB)
- Redirigir temporales de sistema y TensorFlow (TMPDIR) a la carpeta puente
- Ejecutar el script de optimizacion desde esta carpeta puente
- Consejo: mientras compile, abre otra terminal y pon watch df -h para ver cómo se va llenando ese Disco 2 en tiempo real. 
- ADICIONAL: para tener acceso al dataset de nuscenes crear segundo puente

Mi sistema:
    RAM: 32 GB 
    VRAM: 16 GB
    DISK1: 140 GB
    DISK2: 420 GB

FUERA DEL DOCKER: 


``` bash
    # Configurar el puente físico entre discos
    ORIGEN="/home/inartrans2/Documents/INES_TFM/docker_oficial_hailo/shared_with_docker/disco2_puente"
    DESTINO="/media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e87/INES/TFM/hailo_temp"

    mkdir -p $DESTINO
    sudo mount --bind $DESTINO $ORIGEN
    sudo chmod -R 777 $DESTINO
    echo "✅ Puente activado: $ORIGEN -> $DESTINO"
```
``` bash
    # Definir las rutas  6c8f6887-9acf-4794-b990-8de964c59e87
    DATASET_REAL="/media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e87/INES/TFM/datasets/raw/nuscenes/v1.0-trainval"
    PUNTO_ACCESO_DOCKER="/home/inartrans2/Documents/INES_TFM/docker_oficial_hailo/shared_with_docker/hailo-compilation-workflow/data/nuscenes/v1.0-trainval"

    # Crear la carpeta de destino si no existe
    mkdir -p $PUNTO_ACCESO_DOCKER

    # Crear el puente
    sudo mount --bind $DATASET_REAL $PUNTO_ACCESO_DOCKER
    sudo chmod -R 777 $PUNTO_ACCESO_DOCKER

    echo "✅ Segundo puente activado: Dataset NuScenes vinculado."
```

DENTRO DEL DOCKER:
    1. Restart container

    ``` bash
        cd ~/path/docker_oficial_hailo
        sudo rm -r /tmp/hailo_docker.xauth
        ./hailo_ai_sw_suite_docker_run.sh --resume
    ```
    
    2. Ejecutar desde el disco2_puente

    ```bash
        # 1. Redirigir temporales de sistema y TensorFlow
        export TMPDIR=/local/shared_with_docker/disco2_puente

        # 2. Situarse en la lanzadera del disco grande
        cd /local/shared_with_docker/disco2_puente

        # 3. Lanzar optimización (ejemplo con rutas absolutas)
        python /local/shared_with_docker/hailo-compilation-workflow/pillarnest_scripts/3_quantization_pillarnest.py \
            --har /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/modelo.har \
            --calib /local/shared_with_docker/hailo-compilation-workflow/data/calib.npy \
            --alls /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/config.alls
    
        python /local/shared_with_docker/hailo-compilation-workflow/pillarnest_scripts/3_quantization_pillarnest.py \
        --har /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_original/pillarnest_tiny_hailo.har \
        --output /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_original/pillarnest_tiny_hailo_opt12.q.har \
        --calib /local/shared_with_docker/hailo-compilation-workflow/data/calib/calib_dataset_dens25.npy \
        --alls /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_original/optimization12.alls
    
 python /local/shared_with_docker/hailo-compilation-workflow/pillarnest_scripts/3_quantization_clean.py \
        --har /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_small/pillarnest_small.har \
        --output /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_small/pillarnest_small_opt4.q.har \
        --calib /local/shared_with_docker/hailo-compilation-workflow/data/calib/calib_dataset_dens25_128.npy \
        --alls /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_small/opt4.alls

export CUDA_VISIBLE_DEVICES=""
     python /local/shared_with_docker/hailo-compilation-workflow/pillarnest_scripts/3_quantization_pillarnest.py \
        --har /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_tiny_def/pillarnest_tiny_original_export.har \
        --output /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_tiny_def/pillarnest_tiny_original_export_opt4.q.har \
        --calib /local/shared_with_docker/hailo-compilation-workflow/data//nuscenes/v1.0-trainval/calib/calib_dataset_dens25.npy \
        --alls /local/shared_with_docker/hailo-compilation-workflow/model/hailo8l/pillarnest_tiny_def/opt4.alls
    ```


## ESTRUCTURA CARPETAS 

```text
(hailo_virtualenv) hailo@inartrans2:/local/shared_with_docker/hailo-compilation-workflow

hailo-compilation-workflow
    ├── basic_workflow_scripts
    ├── cfgs
    ├── checkpoints
    ├── data/                          
        ├── custom/                 
        ├── kitti/
        ├── waymo/ 
        ├── nuscenes/   
    ├── inference_output/                     
    ├── logs/                 
    ├── model/                              
    ├── openpcdet_scripts/
    ├── output/      
    ├── src/
        ├── pcdet/      # paquete de openpcdet pcdet-0.6.0+0              

```


Si se va a usar algun modelo de OpenPCDet es necesario el paquete pcdet: 
``` bash
pip install -v -e . --no-build-isolation
pip install spconv-cu120	
```

## COMPILATION WORKFLOW

Contexto del Flujo de Trabajo (Hailo Basic Workflow):
    Paso 0: Preparación del modelo -> Exportación a formato intermedio (onnx, tflite).
    Paso 1: Parseo (translation) -> Convierte a formato Hailo Archive (.har). Analiza si las operaciones/activaciones se admiten en el hailo [SUPPORTED LAYERS](https://www.notion.so/SUPPORTED-LAYERS-1aee4a4e7fe38031bb94eb385315142a?pvs=21).
    Paso 2: Optimización -> Cuantización (INT8) usando datos de calibración, también se pueden aplicar distintas optimizaciones.
    Paso 3: Compilación -> Generación del binario .hef para el chip Hailo-8/8L.

Notas Técnicas para Hailo:
    1. Formato de Entrada: Para modelos TensorFlow 2.x, Hailo recomienda encarecidamente
       usar TFLite como punto de entrada para el parser.
    2. Precisión: Este script exporta el modelo en Float32. NO aplicamos cuantización
       de TFLite aquí. La cuantización a INT8/INT4 se realiza dentro del Hailo DFC 
       durante la fase de 'Optimization' para aprovechar los algoritmos avanzados 
       de Hailo (como corrección de bias y ecualización)[cite: 9117, 9122, 9123].
    3. Input Shape: El hardware de Hailo requiere dimensiones de entrada estáticas.
       Asegúrate de que el modelo no tenga dimensiones dinámicas (None) antes de exportar.
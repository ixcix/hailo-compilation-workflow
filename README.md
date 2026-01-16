
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
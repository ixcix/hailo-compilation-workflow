import numpy as np

class PointPillarsConfig:

    use_numba = True

    # =========================================================
    # 1. ENTRADA Y VOXELIZACIÓN
    # =========================================================
    # Límites del entorno 3D [x_min, y_min, z_min, x_max, y_max, z_max]
    point_cloud_range = [-50, -50, -5, 50, 50, 3]

    # Tamaño de cada voxel en metros [x, y, z]
    voxel_size = [0.25, 0.25, 8]

    # Grid size calculado para el Scatter: (100 / 0.25 = 400)
    grid_size = [400, 400, 1]

    # Máximo número de voxeles permitidos [Train, Inference]
    # Para Hailo, el pre-procesado debe ser determinista, usaremos el valor de inferencia.
    max_voxels = 40000

    # Máximo número de puntos por voxel (para reducir densidad)
    max_num_points = 64

    # Dimensiones de entrada (x, y, z, intensidad, timestamp)
    num_input_features = 5

    # Número de barridos temporales (lidar sweeps) acumulados
    sweeps_num = 10

    # =========================================================
    # 2. ARQUITECTURA DEL MODELO
    # =========================================================
    
    # revisar esto puede q algunos sean innecesarios o no se usen 
    # HardVFE
    in_channels_vfe=4
    feat_channels_vfe =[64, 64]  
    norm_cfg=dict(type='naiveSyncBN1d', eps=0.001, momentum=0.01) # no se que es esto pero en la config oficial de pointpillars esta 
    # SCATTER
    in_channels_scatter=64
    output_shape=[400, 400]

    # =========================================================
    # 3. DEFINICIÓN DE TAREAS (Cabezales de Detección)
    # =========================================================
    # Configuración Multi-Head de CenterPoint.
    # El orden en esta lista es CRÍTICO: define el orden de salida de los tensores.
    tasks = [
        dict(num_class=1, class_names=['car']),
        dict(num_class=2, class_names=['truck', 'construction_vehicle']),
        dict(num_class=2, class_names=['bus', 'trailer']),
        dict(num_class=1, class_names=['barrier']),
        dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
    ]

    # Lista plana de todas las clases para referencia rápida
    class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]


    # Parámetros comunes del Anchor3DHead
    nms_type = 'rotate_aprox'  # Tipo de NMS: 'rotate_exact' o 'rotate_approx'
    num_classes=10
    dir_offset=0.7854
    dir_limit_offset=0
    nms_pre=1000
    nms_thr=0.2
    score_thr=0.05
    max_num=500

    anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795],
                    [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
                    [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
                    [-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111],
                    [-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072],
                    [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
                    [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965]],
            sizes=[[1.95017717, 4.60718145, 1.72270761],
                   [2.4560939, 6.73778078, 2.73004906],
                   [2.87427237, 12.01320693, 3.81509561],
                   [0.60058911, 1.68452161, 1.27192197],
                   [0.66344886, 0.7256437, 1.75748069],
                   [0.39694519, 0.40359262, 1.06232151],
                   [2.49008838, 0.48578221, 0.98297065]],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True)
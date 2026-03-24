import numpy as np

class PillarnestTinyConfig:
    """
    Configuración centralizada para PillarNest (Versión Tiny).
    Diseñada para ser consumida por scripts agnósticos de Pre y Post procesado.
    """
    use_numba = True  
    # =========================================================
    # 1. ENTRADA Y VOXELIZACIÓN 
    # =========================================================
    # Límites del entorno 3D [x_min, y_min, z_min, x_max, y_max, z_max]
    point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    
    # Tamaño de cada voxel en metros [x, y, z]
    voxel_size = [0.15, 0.15, 8.0]
    
    # Resolución de la rejilla resultante [W, H, D]
    # Calculado como: (max - min) / voxel_size. 
    # (108 / 0.15 = 720). Importante para el Scatter.
    grid_size = [720, 720, 1]
    
    # Máximo número de voxeles permitidos [Train, Inference]
    # Para Hailo, el pre-procesado debe ser determinista, usaremos el valor de inferencia.
    max_voxels = [30000, 40000]
    
    # Máximo número de puntos por voxel (para reducir densidad)
    max_num_points = 20
    
    # Dimensiones de entrada (x, y, z, intensidad, timestamp)
    num_input_features = 5
    
    # Número de barridos temporales (lidar sweeps) acumulados
    sweeps_num = 9


    # =========================================================
    # 2. ARQUITECTURA DEL MODELO 
    # =========================================================
    # Factor de reducción espacial del Backbone.
    # Si Input=720x720 y Stride=4 -> Output=180x180.
    out_size_factor = 4
    
    # Canales de salida del codificador de pilares (Input del Backbone 2D)
    feat_channels = 48


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
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]


    # =========================================================
    # 4. POST-PROCESADO Y DECODING 
    # =========================================================
    # --- Filtrado Básico ---
    # Puntuación mínima para considerar una detección válida
    score_threshold = 0.1
    
    # Rango límite de centros (ligeramente mayor que point_cloud_range)
    # Se usa para descartar predicciones erróneas en los bordes.
    post_center_limit_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

    # --- Top-K y Límites ---
    # Cuántos candidatos extraer del mapa de calor antes de NMS
    pre_max_size = 1000
    
    # Cuántas detecciones finales devolver como máximo
    post_max_size = 83

    # --- NMS (Non-Maximum Suppression) ---
    # Tipo de NMS: 'rotate' (más preciso, lento) o 'circle' (rápido por distancia)
    nms_type = 'circle'
    
    # Umbral de solapamiento (IoU) para fusionar cajas (0.2 es típico en LiDAR)
    nms_thr = 0.2
    
    # Radios específicos para 'circle_nms' (uno por tarea). 
    # Se usa si nms_type='circle'.
    min_radius = [4, 12, 10, 1, 0.85, 0.175]

    # --- Rectificación de Score ---
    # Factor de mezcla entre Heatmap Score y IoU Prediction.
    # Final Score = (heatmap ^ (1-beta)) * (iou ^ beta)
    iou_score_beta = 0.5



class PillarnestSmallConfig:
    """
    Configuración centralizada para PillarNest (Versión Small).
    Diseñada para ser consumida por scripts agnósticos de Pre y Post procesado.
    """

    # =========================================================
    # 1. ENTRADA Y VOXELIZACIÓN 
    # =========================================================
    # Límites del entorno 3D [x_min, y_min, z_min, x_max, y_max, z_max]
    point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    
    # Tamaño de cada voxel en metros [x, y, z]
    voxel_size = [0.15, 0.15, 8.0]
    
    # Resolución de la rejilla resultante [W, H, D]
    # Calculado como: (max - min) / voxel_size. 
    # (108 / 0.15 = 720). Importante para el Scatter.
    grid_size = [720, 720, 1]
    
    # Máximo número de voxeles permitidos [Train, Inference]
    # Para Hailo, el pre-procesado debe ser determinista, usaremos el valor de inferencia.
    max_voxels = [30000, 40000]
    
    # Máximo número de puntos por voxel (para reducir densidad)
    max_num_points = 20
    
    # Dimensiones de entrada (x, y, z, intensidad, timestamp)
    num_input_features = 5
    
    # Número de barridos temporales (lidar sweeps) acumulados
    sweeps_num = 9


    # =========================================================
    # 2. ARQUITECTURA DEL MODELO 
    # =========================================================
    # Factor de reducción espacial del Backbone.
    # Si Input=720x720 y Stride=4 -> Output=180x180.
    out_size_factor = 4
    
    # Canales de salida del codificador de pilares (Input del Backbone 2D)
    feat_channels = 48


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
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]


    # =========================================================
    # 4. POST-PROCESADO Y DECODING 
    # =========================================================
    # --- Filtrado Básico ---
    # Puntuación mínima para considerar una detección válida
    score_threshold = 0.1
    
    # Rango límite de centros (ligeramente mayor que point_cloud_range)
    # Se usa para descartar predicciones erróneas en los bordes.
    post_center_limit_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

    # --- Top-K y Límites ---
    # Cuántos candidatos extraer del mapa de calor antes de NMS
    pre_max_size = 1000
    
    # Cuántas detecciones finales devolver como máximo
    post_max_size = 83

    # --- NMS (Non-Maximum Suppression) ---
    # Tipo de NMS: 'rotate' (más preciso, lento) o 'circle' (rápido por distancia)
    nms_type = 'rotate'
    
    # Umbral de solapamiento (IoU) para fusionar cajas (0.2 es típico en LiDAR)
    nms_thr = 0.2
    
    # Radios específicos para 'circle_nms' (uno por tarea). 
    # Se usa si nms_type='circle'.
    min_radius = [4, 12, 10, 1, 0.85, 0.175]

    # --- Rectificación de Score ---
    # Factor de mezcla entre Heatmap Score y IoU Prediction.
    # Final Score = (heatmap ^ (1-beta)) * (iou ^ beta)
    iou_score_beta = 0.5
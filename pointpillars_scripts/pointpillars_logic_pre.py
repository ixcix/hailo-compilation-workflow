import numpy as np
import os
import numba

class Loader:
    def __init__(self, load_dim=5, use_dim=[0, 1, 2, 3]):
        """
        Cargador agnóstico: solo sabe cuántas dimensiones leer y cuáles usar.
        """
        self.load_dim = load_dim
        self.use_dim = use_dim 

    def load_points(self, pts_filename):
        if not os.path.exists(pts_filename):
            raise FileNotFoundError(f"Archivo no encontrado: {pts_filename}")

        # Lectura eficiente de binarios
        points = np.fromfile(pts_filename, dtype=np.float32).reshape(-1, self.load_dim)
        points = points[:, self.use_dim]

        # Preparamos la 5ª columna para el tiempo (MultiSweep)
        points_with_time = np.zeros((points.shape[0], 5), dtype=np.float32)
        points_with_time[:, :4] = points
        
        return points_with_time

class MultiSweep:
    def __init__(self, 
                 sweeps_num=9, 
                 load_dim=5, 
                 use_dim=[0, 1, 2, 3, 4], 
                 pad_empty_sweeps=True, 
                 remove_close=True, 
                 test_mode=True):
        """
        Lógica de acumulación agnóstica: recibe los parámetros de sweeps y filtrado.
        """
        self.sweeps_num = sweeps_num
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _remove_close(self, points, radius=1.0):
        x_filt = np.abs(points[:, 0]) < radius
        y_filt = np.abs(points[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def process(self, results):
        points = results['points']
        points[:, 4] = 0 # El frame actual tiene lag 0
        
        sweep_points_list = [points]
        ts = results['timestamp'] # Segundos

        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                p_copy = np.copy(points)
                if self.remove_close:
                    p_copy = self._remove_close(p_copy)
                sweep_points_list.append(p_copy)
        else:
            # Determinamos cuántos sweeps procesar (test_mode suele ser True en inferencia)
            num_to_process = self.sweeps_num if self.test_mode else len(results['sweeps'])
            choices = np.arange(min(len(results['sweeps']), num_to_process))

            for idx in choices:
                sweep = results['sweeps'][idx]
                
                # Carga de sweep pasado
                points_sweep = np.fromfile(sweep['data_path'], dtype=np.float32).reshape(-1, self.load_dim)
                points_sweep = np.copy(points_sweep)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)

                # Alineación temporal
                sweep_ts = sweep['timestamp'] / 1e6
                
                # Transformación geométrica (proyectar pasado al presente)
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                
                # Lag temporal en la 5ª columna
                points_sweep[:, 4] = ts - sweep_ts
                sweep_points_list.append(points_sweep)

        # Unión y filtrado de dimensiones final
        all_points = np.concatenate(sweep_points_list, axis=0)
        results['points'] = all_points[:, self.use_dim]
        return results
    

# --- FUNCIÓN NÚCLEO EN C (FUERA DE LA CLASE) ---
@numba.jit(nopython=True)
def _voxelize_numba_core(points, voxel_size, pcd_range, grid_size, max_num_points, max_voxels):
    """
    Algoritmo de voxelización de 1 sola pasada O(N).
    Extremadamente rápido al usar un grid de acceso directo O(1).
    """
    N, C = points.shape
    grid_x, grid_y, grid_z = grid_size[0], grid_size[1], grid_size[2]
    
    # Grid denso para mapear coordenadas -> Voxel_ID instantáneamente
    grid_volume = grid_x * grid_y * grid_z
    coor_to_voxelidx = np.full(grid_volume, -1, dtype=np.int32)
    
    voxels = np.zeros((max_voxels, max_num_points, C), dtype=np.float32)
    coors = np.zeros((max_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(max_voxels, dtype=np.int32)
    
    voxel_num = 0
    
    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        
        # 1. Filtro de rango espacial
        if not (pcd_range[0] <= x < pcd_range[3] and
                pcd_range[1] <= y < pcd_range[4] and
                pcd_range[2] <= z < pcd_range[5]):
            continue
            
        # 2. Coordenadas del grid
        c_x = int(np.floor((x - pcd_range[0]) / voxel_size[0]))
        c_y = int(np.floor((y - pcd_range[1]) / voxel_size[1]))
        c_z = int(np.floor((z - pcd_range[2]) / voxel_size[2]))
        
        # Filtro de seguridad
        if c_x < 0 or c_x >= grid_x or c_y < 0 or c_y >= grid_y or c_z < 0 or c_z >= grid_z:
            continue
        
        # 3. Índice plano O(1)
        idx = c_z * (grid_y * grid_x) + c_y * grid_x + c_x
        voxel_idx = coor_to_voxelidx[idx]
        
        # 4. Creación de pilar nuevo
        if voxel_idx == -1:
            if voxel_num >= max_voxels:
                continue
            voxel_idx = voxel_num
            coor_to_voxelidx[idx] = voxel_idx
            coors[voxel_idx, 0] = c_z
            coors[voxel_idx, 1] = c_y
            coors[voxel_idx, 2] = c_x
            voxel_num += 1
        
        # 5. Adición de punto al pilar
        pts_in_voxel = num_points_per_voxel[voxel_idx]
        if pts_in_voxel < max_num_points:
            voxels[voxel_idx, pts_in_voxel, :] = points[i]
            num_points_per_voxel[voxel_idx] += 1
            
    return voxels[:voxel_num], coors[:voxel_num], num_points_per_voxel[:voxel_num]
# --- NÚCLEO EN C PARA EL ENCODER (PointPillars HardVFE) ---
@numba.njit(fastmath=True, cache=True)
def _vfe_numba_core(voxels, num_points, coors, P,
                    vx, vy, vz, x_offset, y_offset, z_offset, 
                    w0, b0, w1, b1):
    M = voxels.shape[0]
    out_features = np.zeros((M, 64), dtype=np.float32)
    
    for i in range(M):
        n_pts = num_points[i]
        if n_pts == 0:
            continue
            
        # 1. Feature Decoration: Cluster Center
        sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
        for j in range(n_pts):
            sum_x += voxels[i, j, 0]
            sum_y += voxels[i, j, 1]
            sum_z += voxels[i, j, 2]
        mean_x = sum_x / n_pts
        mean_y = sum_y / n_pts
        mean_z = sum_z / n_pts
        
        # 2. Feature Decoration: Voxel Center
        c_z = coors[i, 0] * vz + z_offset
        c_y = coors[i, 1] * vy + y_offset
        c_x = coors[i, 2] * vx + x_offset
        
        # Buffers locales para la Capa 0
        layer0_out = np.zeros((n_pts, 64), dtype=np.float32)
        max_pool_0 = np.zeros(64, dtype=np.float32)
        
        # --- VFE LAYER 0 ---
        for j in range(n_pts):
            f0 = voxels[i, j, 0]
            f1 = voxels[i, j, 1]
            f2 = voxels[i, j, 2]
            f3 = voxels[i, j, 3] # Intensity (4º canal, ignoramos el tiempo que es el 5º)
            f4 = f0 - mean_x
            f5 = f1 - mean_y
            f6 = f2 - mean_z
            f7 = f0 - c_x
            f8 = f1 - c_y
            f9 = f2 - c_z
            
            for k in range(64):
                val = (f0*w0[k,0] + f1*w0[k,1] + f2*w0[k,2] + f3*w0[k,3] +
                       f4*w0[k,4] + f5*w0[k,5] + f6*w0[k,6] + f7*w0[k,7] +
                       f8*w0[k,8] + f9*w0[k,9] + b0[k])
                
                if val < 0.0:
                    val = 0.0 # ReLU
                
                layer0_out[j, k] = val
                if val > max_pool_0[k]:
                    max_pool_0[k] = val
                    
        # --- VFE LAYER 1 + POOLING FINAL ---
        for j in range(n_pts):
            for k in range(64):
                val = b1[k]
                # Características punto a punto de la capa 0
                for c in range(64):
                    val += layer0_out[j, c] * w1[k, c]
                # Max pooling global de la capa 0 concatenado
                for c in range(64):
                    val += max_pool_0[c] * w1[k, c + 64]
                    
                if val < 0.0:
                    val = 0.0 # ReLU
                    
                if val > out_features[i, k]:
                    out_features[i, k] = val
        
        # --- EMULACIÓN EXACTA DEL PADDING (Para mantener el Bit-Exact) ---
        # Si el pilar tiene menos puntos que el max_num_points (P), la implementación
        # NumPy pasaba ceros a la Capa 1, resultando en max(b1, 0). 
        # Hay que incluirlo en el Max Pooling final para no perder la precisión del modelo oficial.
        if n_pts < P:
            for k in range(64):
                pad_val = b1[k]
                if pad_val < 0.0:
                    pad_val = 0.0
                if pad_val > out_features[i, k]:
                    out_features[i, k] = pad_val
                    
    return out_features


class Voxelizer:
    def __init__(self,
                 voxel_size, 
                 point_cloud_range, 
                 max_num_points, 
                 max_voxels):
        """
        Voxelizador agnóstico: recibe dimensiones y límites por parámetro.
        """

        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels

        # Cálculo dinámico del grid (720x720 para tu config actual)
        grid_size = (self.pcd_range[3:] - self.pcd_range[:3]) / self.voxel_size
        self.grid_size = np.round(grid_size).astype(np.int64)

    # # Versión oficial con bucle (lento en CPU)
    # def voxelize(self, points):
    #     # 1. Filtro de rango espacial
    #     mask = ((points[:, 0] >= self.pcd_range[0]) & (points[:, 0] < self.pcd_range[3]) &
    #             (points[:, 1] >= self.pcd_range[1]) & (points[:, 1] < self.pcd_range[4]) &
    #             (points[:, 2] >= self.pcd_range[2]) & (points[:, 2] < self.pcd_range[5]))
    #     points = points[mask]

    #     if len(points) == 0:
    #         return None

    #     # 2. Coordenadas de Voxel (Z, Y, X)
    #     voxel_coords = ((points[:, :3] - self.pcd_range[:3]) / self.voxel_size).astype(np.int32)
    #     voxel_coords = voxel_coords[:, [2, 1, 0]] 

    #     # 3. Hashing para agrupación rápida sin bucles
    #     pilar_id = (voxel_coords[:, 0] * self.grid_size[1] * self.grid_size[0] +
    #                 voxel_coords[:, 1] * self.grid_size[0] +
    #                 voxel_coords[:, 2])

    #     unique_ids, first_point_indices, inverse_indices = np.unique(
    #         pilar_id, return_index=True, return_inverse=True
    #     )

    #     # 4. Limitación de Voxels (Consistencia con memoria del Hailo)
    #     num_voxels = min(len(unique_ids), self.max_voxels)
    #     keep_points_mask = inverse_indices < num_voxels
    #     points = points[keep_points_mask]
    #     inverse_indices = inverse_indices[keep_points_mask]

    #     # 5. Llenado de tensores finales
    #     voxels = np.zeros((num_voxels, self.max_num_points, points.shape[1]), dtype=np.float32)
    #     coors = voxel_coords[first_point_indices[:num_voxels]]
    #     num_points_per_voxel = np.zeros((num_voxels,), dtype=np.int32)

    #     for i in range(num_voxels):
    #         curr_points = points[inverse_indices == i]
    #         n_pts = min(len(curr_points), self.max_num_points)
    #         voxels[i, :n_pts, :] = curr_points[:n_pts]
    #         num_points_per_voxel[i] = n_pts

    #     return voxels, coors, num_points_per_voxel

    # Esta versión evita el bucle for de 30.000 iteraciones usando indexación avanzada (cambia por tanto el orden de los ptos dentro del pilar)
    def voxelize(self, points):
        # 1. Filtro de rango (Sigue igual, es rápido)
        mask = ((points[:, 0] >= self.pcd_range[0]) & (points[:, 0] < self.pcd_range[3]) &
                (points[:, 1] >= self.pcd_range[1]) & (points[:, 1] < self.pcd_range[4]) &
                (points[:, 2] >= self.pcd_range[2]) & (points[:, 2] < self.pcd_range[5]))
        points = points[mask]

        if len(points) == 0:
            return None

        # 2. Coordenadas de Voxel (Z, Y, X)
        voxel_coords = ((points[:, :3] - self.pcd_range[:3]) / self.voxel_size).astype(np.int32)
        voxel_coords = voxel_coords[:, [2, 1, 0]] 

        # 3. ID Único para Hashing
        pilar_id = (voxel_coords[:, 0] * self.grid_size[1] * self.grid_size[0] +
                    voxel_coords[:, 1] * self.grid_size[0] +
                    voxel_coords[:, 2])

        # 4. Agrupación Vectorizada
        # Obtenemos los índices que ordenarían los puntos por pilar_id
        sort_idx = np.argsort(pilar_id)
        pilar_id_sorted = pilar_id[sort_idx]
        points_sorted = points[sort_idx]
        coords_sorted = voxel_coords[sort_idx]

        # Identificamos dónde cambia cada pilar
        unique_ids, first_indices, counts = np.unique(pilar_id_sorted, return_index=True, return_counts=True)

        # 5. Limitación de Voxels
        num_voxels = min(len(unique_ids), self.max_voxels)
        unique_ids = unique_ids[:num_voxels]
        first_indices = first_indices[:num_voxels]
        counts = counts[:num_voxels]

        # 6. Llenado ultra-rápido sin bucles pesados
        voxels = np.zeros((num_voxels, self.max_num_points, points.shape[1]), dtype=np.float32)
        coors = coords_sorted[first_indices]
        num_points_per_voxel = np.minimum(counts, self.max_num_points).astype(np.int32)

        # Creamos una máscara para coger solo hasta max_num_points de cada pilar
        # Esto evita el bucle for de 30.000 iteraciones
        for i in range(self.max_num_points):
            # Seleccionamos el i-ésimo punto de cada pilar (si existe)
            valid_pilar_mask = counts > i
            if not np.any(valid_pilar_mask):
                break
            
            # El índice del i-ésimo punto en el array ordenado es first_index + i
            global_indices = first_indices[valid_pilar_mask] + i
            voxels[valid_pilar_mask, i, :] = points_sorted[global_indices]

        return voxels, coors, num_points_per_voxel
    
    # Version que usa numba
    def voxelize_numba(self, points):
        """
        Función pública inalterada. Extrae los parámetros y llama al núcleo en C.
        """
        if len(points) == 0:
            return None
            
        return _voxelize_numba_core(
            points=points,
            voxel_size=self.voxel_size,
            pcd_range=self.pcd_range,
            grid_size=self.grid_size,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels
        )
    
class PointpillarsHardVFE:
    def __init__(self, 
                 voxel_size=[0.25, 0.25, 8],
                 point_cloud_range=[-50, -50, -5, 50, 50, 3]):
        
        self.vx, self.vy, self.vz = voxel_size
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        
        self.w0 = None
        self.b0 = None
        self.w1 = None
        self.b1 = None

    def set_weights(self, weights_dict):
        """ Fusión Linear + BN idéntica a tu implementación NumPy """
        fused = {}
        for i in range(2):
            prefix = f'pts_voxel_encoder.vfe_layers.{i}'
            w = weights_dict[f'{prefix}.linear.weight']
            b = weights_dict.get(f'{prefix}.linear.bias', np.zeros(w.shape[0], dtype=np.float32))
            
            gamma = weights_dict[f'{prefix}.norm.weight']
            beta = weights_dict[f'{prefix}.norm.bias']
            mean = weights_dict[f'{prefix}.norm.running_mean']
            var = weights_dict[f'{prefix}.norm.running_var']
            eps = 1e-3

            scale = gamma / np.sqrt(var + eps)
            fused[i] = {
                'w': (w * scale[:, None]).astype(np.float32),
                'b': ((b - mean) * scale + beta).astype(np.float32)
            }
            
        # Numba requiere arrays contiguos para máxima velocidad, no soporta diccionarios
        self.w0 = np.ascontiguousarray(fused[0]['w'])
        self.b0 = np.ascontiguousarray(fused[0]['b'])
        self.w1 = np.ascontiguousarray(fused[1]['w'])
        self.b1 = np.ascontiguousarray(fused[1]['b'])

    def encode(self, voxels, num_points, coors):
        # voxels: (M, P, 5) -> Pasamos la P (max_points) al core
        P = voxels.shape[1]
        
        return _vfe_numba_core(
            voxels, num_points, coors, P,
            self.vx, self.vy, self.vz, 
            self.x_offset, self.y_offset, self.z_offset,
            self.w0, self.b0, self.w1, self.b1
        )


# # Implementacion del scatter oficial con output NCHW (1, 48, 720, 720)
# class Scatter:
#     def __init__(self, 
#                  output_shape=[720, 720], 
#                  num_input_features=48):
#         """
#         Bloque Scatter: Convierte la lista dispersa de pilares en una imagen densa.
#         Equivalente a: mmdet3d.models.middle_encoders.PointPillarsScatter
#         """
#         # output_shape suele ser [720, 720] en tu config
#         self.ny = output_shape[0] # Alto (H)
#         self.nx = output_shape[1] # Ancho (W)
#         self.in_channels = num_input_features
        
#         # Buffer opcional para optimización futura
#         self.canvas_buffer = None

#     def scatter(self, voxel_features, coors):
#         """
#         Args:
#             voxel_features: (M, 48) - Features calculados por el Encoder.
#             coors: (M, 3) - Coordenadas [z, y, x] de cada pilar.
            
#         Returns:
#             canvas: (48, 720, 720) - Imagen pseudo-lidar FP32.
#         """
#         # 1. Crear lienzo vacío (FP32)
#         # Formato NCHW estándar de PyTorch: (Canales, Alto, Ancho)
#         # Allocating ~99 MB cada frame (48 * 720 * 720 * 4 bytes)
#         canvas = np.zeros((self.in_channels, self.ny, self.nx), dtype=np.float32)

#         # 2. Extraer índices Y, X
#         # En tu Voxelizer coors es [z, y, x].
#         # coors[:, 1] es Y
#         # coors[:, 2] es X
#         y_idxs = coors[:, 1]
#         x_idxs = coors[:, 2]

#         # 3. Filtrado de Seguridad (Clip)
#         # El código oficial asume que los índices están bien, pero en producción
#         # es vital asegurar que no escribimos fuera del array.
#         mask = (y_idxs >= 0) & (y_idxs < self.ny) & (x_idxs >= 0) & (x_idxs < self.nx)
        
#         # Si todos son válidos (lo normal), pasamos directo. Si no, filtramos.
#         if not np.all(mask):
#             voxel_features = voxel_features[mask]
#             y_idxs = y_idxs[mask]
#             x_idxs = x_idxs[mask]

#         # 4. Operación Scatter (Fancy Indexing)
#         # PyTorch hace: canvas_flat[indices] = voxels.t()
#         # NumPy hace: canvas[:, y, x] = features.T
        
#         # voxel_features es (M, 48). Transponemos a (48, M) para encajar en el slice (C, ...)
#         canvas[:, y_idxs, x_idxs] = voxel_features.T
    
#         return canvas


# # Implementación del scatter con output NHWC (1, 720, 720, 48) HAILO-FRIENDLY 
class Scatter:
    def __init__(self, 
                 output_shape=[720, 720], 
                 num_input_features=48):
        """
        Bloque Scatter HAILO-FRIENDLY (Stateless):
        1. Salida (720, 720, 48) -> Formato NHWC nativo para Hailo (evita transposiciones).
        2. Sin Buffer persistente -> Usa np.zeros en cada frame (más rápido en tu prueba).
        """
        self.ny = output_shape[0] # Alto (H)
        self.nx = output_shape[1] # Ancho (W)
        self.in_channels = num_input_features

    def scatter(self, voxel_features, coors):
        """
        Args:
            voxel_features: (M, 48) - Features.
            coors: (M, 3) - Coordenadas [z, y, x].
            
        Returns:
            canvas: (720, 720, 48) - Imagen pseudo-lidar FP32 (NHWC).
        """
        # 1. Crear lienzo nuevo en cada frame (NHWC)
        # Shape: (720, 720, 48). 
        # Al ser NHWC, los 48 canales de un píxel están contiguos en memoria.
        canvas = np.zeros((self.ny, self.nx, self.in_channels), dtype=np.float32)

        # 2. Extraer índices Y, X
        y_idxs = coors[:, 1]
        x_idxs = coors[:, 2]

        # 3. Filtrado de Seguridad (Clip)
        mask = (y_idxs >= 0) & (y_idxs < self.ny) & (x_idxs >= 0) & (x_idxs < self.nx)
        
        if not np.all(mask):
            voxel_features = voxel_features[mask]
            y_idxs = y_idxs[mask]
            x_idxs = x_idxs[mask]

        # 4. Operación Scatter Directa (NHWC)
        # voxel_features es (M, 48).
        # canvas[y, x] espera un vector de (48,).
        # Al coincidir la última dimensión, NumPy hace la copia directa (memcpy)
        # sin necesidad de transponer (.T). Esto es muy eficiente.
        canvas[y_idxs, x_idxs] = voxel_features

        return canvas
    
# # # Implementación del scatter con output NHWC (1, 720, 720, 48) HAILO-FRIENDLY y con opción de cuantización INT8 (experimental)
# class Scatter:
#     def __init__(self, 
#                  output_shape=[720, 720], 
#                  num_input_features=48,
#                  quantize_mode=False,
#                  # Valores obtenidos del análisis estadístico (P99.99%)
#                  quant_scale=0.53099, 
#                  quant_zero_point=0):
#         """
#         Bloque Scatter HAILO-FRIENDLY (Stateless):
#         1. Salida (720, 720, 48) -> Formato NHWC nativo para Hailo.
#         2. Soporta cuantización opcional (experimental).
#         """
#         self.ny = output_shape[0] # Alto (H)
#         self.nx = output_shape[1] # Ancho (W)
#         self.in_channels = num_input_features
        
#         self.quantize_mode = quantize_mode
#         self.scale = quant_scale
#         self.zp = quant_zero_point
        
#         # Pre-cálculo para velocidad
#         if self.scale != 0:
#             self.inv_scale = 1.0 / self.scale
#         else:
#             self.inv_scale = 1.0

#     def scatter(self, voxel_features, coors):
#         """
#         Args:
#             voxel_features: (M, 48) FP32.
#             coors: (M, 3) [z, y, x].
            
#         Returns:
#             canvas: (720, 720, 48) FP32 o UINT8 (NHWC).
#         """
#         # 1. Preparar Índices
#         y_idxs = coors[:, 1]
#         x_idxs = coors[:, 2]

#         # Clip de seguridad
#         mask = (y_idxs >= 0) & (y_idxs < self.ny) & (x_idxs >= 0) & (x_idxs < self.nx)
#         if not np.all(mask):
#             voxel_features = voxel_features[mask]
#             y_idxs = y_idxs[mask]
#             x_idxs = x_idxs[mask]

#         if self.quantize_mode:
#             # === MODO INT8 (Ahorro de Ancho de Banda) ===
#             # Creamos canvas uint8 lleno con el Zero Point
#             canvas = np.full((self.ny, self.nx, self.in_channels), self.zp, dtype=np.uint8)
            
#             # Cuantizamos features: q = round(x / scale) + zp
#             feat_quant = np.rint(voxel_features * self.inv_scale) + self.zp
            
#             # Clip 0-255 y cast
#             np.clip(feat_quant, 0, 255, out=feat_quant)
            
#             # Scatter directo (NHWC)
#             canvas[y_idxs, x_idxs] = feat_quant.astype(np.uint8)
#             return canvas

#         else:
#             # === MODO FP32 (Máxima Precisión - RECOMENDADO) ===
#             # Creamos canvas float32 limpio
#             canvas = np.zeros((self.ny, self.nx, self.in_channels), dtype=np.float32)
            
#             # Scatter directo (NHWC)
#             canvas[y_idxs, x_idxs] = voxel_features
#             return canvas
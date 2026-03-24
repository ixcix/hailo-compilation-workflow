import numpy as np
import numba

class CenterPointPostProcessor:
    def __init__(self, config):
        """
        Post-procesador AGNÓSTICO para CenterPoint.
        """
        # --- 1. Extracción de Configuración (Agnóstica) ---
        self.voxel_size = np.array(getattr(config, 'voxel_size', [0.15, 0.15, 8.0]))
        self.pc_range = np.array(getattr(config, 'point_cloud_range', [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]))
        self.out_size_factor = getattr(config, 'out_size_factor', 4)
        self.tasks = getattr(config, 'tasks', [])
        
        self.score_threshold = getattr(config, 'score_threshold', 0.1)
        self.post_center_limit_range = np.array(getattr(config, 'post_center_limit_range', [-1e5]*3 + [1e5]*3))
        self.pre_max_size = getattr(config, 'pre_max_size', 1000)
        self.post_max_size = getattr(config, 'post_max_size', 83)
        self.nms_type = getattr(config, 'nms_type', 'rotate')
        print(f"INFO: NMS Type configurado como '{self.nms_type}'")
        self.nms_thr = getattr(config, 'nms_thr', 0.2)
        self.min_radius = getattr(config, 'min_radius', [2]*len(self.tasks))
        self.iou_score_beta = getattr(config, 'iou_score_beta', 0.5)
        self.headers = ['reg', 'height', 'dim', 'rot', 'vel', 'iou', 'heatmap']
        
        # --- FLAG DE NUMBA ---
        self.use_numba = getattr(config, 'use_numba', False)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _gather_feat(self, feat, ind):
        C = feat.shape[0]
        feat = feat.reshape(C, -1)
        feat = feat.transpose(1, 0)
        feat = feat[ind]
        return feat

    def _topk(self, scores, K):
        cat, height, width = scores.shape
        scores_flat = scores.reshape(-1)
        real_K = min(K, len(scores_flat))
        
        idx = np.argpartition(scores_flat, -real_K)[-real_K:]
        vals = scores_flat[idx]
        sort_idx = np.argsort(-vals)
        topk_inds = idx[sort_idx]
        topk_scores = vals[sort_idx]

        topk_clses = (topk_inds // (height * width)).astype(np.int32)
        topk_inds_spatial = topk_inds % (height * width)
        topk_ys = (topk_inds_spatial // width).astype(np.float32)
        topk_xs = (topk_inds_spatial % width).astype(np.float32)
        
        return topk_scores, topk_inds_spatial, topk_clses, topk_ys, topk_xs

    def bbox_coder_decode(self, heat, rot_sine, rot_cosine, hei, dim, vel, reg=None, iou_scores=None):
        scores, inds, clses, ys, xs = self._topk(heat, K=self.pre_max_size)

        if reg is not None:
            reg = self._gather_feat(reg, inds)
            xs = xs + reg[:, 0]
            ys = ys + reg[:, 1]
        else:
            xs = xs + 0.5
            ys = ys + 0.5

        rot_sine = self._gather_feat(rot_sine, inds)
        rot_cosine = self._gather_feat(rot_cosine, inds)
        rot = np.arctan2(rot_sine, rot_cosine)

        hei = self._gather_feat(hei, inds)
        dim = self._gather_feat(dim, inds)

        final_iou_scores = None
        if iou_scores is not None:
            final_iou_scores = self._gather_feat(iou_scores, inds)[:, 0]

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        xs = xs[:, None]
        ys = ys[:, None]
        
        if vel is None:
            final_box_preds = np.concatenate([xs, ys, hei, dim, rot], axis=1)
        else:
            vel = self._gather_feat(vel, inds)
            final_box_preds = np.concatenate([xs, ys, hei, dim, rot, vel], axis=1)

        return final_box_preds, scores, clses, final_iou_scores

    # =================================================================================
    # IMPLEMENTACIONES NMS: VERSIÓN NUMPY PURA (VECTORIZADA)
    # =================================================================================
    
    def _rotate_nms_numpy(self, boxes, scores, iou_threshold):
        if len(boxes) == 0: return []
        x, y, w, l, rot = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
        
        cos_rot = np.abs(np.cos(rot))
        sin_rot = np.abs(np.sin(rot))
        w_new = w * cos_rot + l * sin_rot
        l_new = w * sin_rot + l * cos_rot
        
        x1 = x - w_new / 2.0
        y1 = y - l_new / 2.0
        x2 = x + w_new / 2.0
        y2 = y + l_new / 2.0
        
        areas = w_new * l_new
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w_inter = np.maximum(0.0, xx2 - xx1)
            h_inter = np.maximum(0.0, yy2 - yy1)
            inter = w_inter * h_inter
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def _circle_nms(self, dets, scores, labels, radius):
        if len(dets) == 0: return []
        x = dets[:, 0]
        y = dets[:, 1]
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Distancia euclídea AL CUADRADO vectorial
            dx = x[order[1:]] - x[i]
            dy = y[order[1:]] - y[i]
            dist_sq = dx**2 + dy**2
            
            inds = np.where(dist_sq > radius)[0]
            order = order[inds + 1]
            
        return keep

    # =================================================================================
    # IMPLEMENTACIONES NMS: VERSIÓN NUMBA OPTIMIZADA (C-STYLE)
    # =================================================================================

    @staticmethod
    @numba.jit(nopython=True)
    def _rotate_nms_numpy_numba(boxes, scores, iou_threshold):
        ndets = boxes.shape[0]
        if ndets == 0: 
            return np.zeros(0, dtype=np.int32) # Devolvemos array tipado vacío
        
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 3]
        l = boxes[:, 4]
        rot = boxes[:, 6]
        
        cos_rot = np.abs(np.cos(rot))
        sin_rot = np.abs(np.sin(rot))
        w_new = w * cos_rot + l * sin_rot
        l_new = w * sin_rot + l * cos_rot
        
        x1 = x - w_new / 2.0
        y1 = y - l_new / 2.0
        x2 = x + w_new / 2.0
        y2 = y + l_new / 2.0
        
        areas = w_new * l_new
        
        order = np.argsort(-scores).astype(np.int32)
        suppressed = np.zeros(ndets, dtype=np.int32)
        
        # En C++ es mucho más rápido crear un array del tamaño máximo y llevar un contador
        keep = np.zeros(ndets, dtype=np.int32)
        count = 0
        
        for _i in range(ndets):
            i = order[_i]
            if suppressed[i] == 1:
                continue
            
            keep[count] = i
            count += 1
            
            for _j in range(_i + 1, ndets):
                j = order[_j]
                if suppressed[j] == 1:
                    continue
                    
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                
                w_inter = max(0.0, xx2 - xx1)
                h_inter = max(0.0, yy2 - yy1)
                inter = w_inter * h_inter
                
                ovr = inter / (areas[i] + areas[j] - inter)
                
                if ovr > iou_threshold:
                    suppressed[j] = 1
                    
        return keep[:count] # Devolvemos solo la parte llena del array

    @staticmethod
    @numba.jit(nopython=True)
    def _circle_nms_numba(dets, scores, labels, radius):
        ndets = dets.shape[0]
        if ndets == 0: 
            return np.zeros(0, dtype=np.int32)

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        order = np.argsort(-scores).astype(np.int32)
        
        suppressed = np.zeros(ndets, dtype=np.int32)
        
        # Mismo truco del array pre-asignado
        keep = np.zeros(ndets, dtype=np.int32)
        count = 0
        
        for _i in range(ndets):
            i = order[_i]
            if suppressed[i] == 1:
                continue
                
            keep[count] = i
            count += 1
            
            for _j in range(_i + 1, ndets):
                j = order[_j]
                if suppressed[j] == 1:
                    continue
                
                dist_sq = (x1[i] - x1[j])**2 + (y1[i] - y1[j])**2
                
                if dist_sq <= radius:
                    suppressed[j] = 1
                    
        return keep[:count]

    # =================================================================================
    # SHAPELY NMS (SOLO CPU/NUMPY, INCOMPATIBLE CON NUMBA)
    # =================================================================================

    def _rotate_nms_exact_cpu(self, boxes, scores, iou_threshold):
        from shapely.geometry import Polygon
        if len(boxes) == 0: return []
        
        polygons = []
        areas = []
        for box in boxes:
            x, y, w, l, rot = box[0], box[1], box[3], box[4], box[6]
            cos_a, sin_a = np.cos(rot), np.sin(rot)
            dx_w, dy_w = (w / 2) * cos_a, (w / 2) * sin_a
            dx_l, dy_l = (l / 2) * -sin_a, (l / 2) * cos_a
            
            p1 = (x + dx_w + dx_l, y + dy_w + dy_l)
            p2 = (x + dx_w - dx_l, y + dy_w - dy_l)
            p3 = (x - dx_w - dx_l, y - dy_w - dy_l)
            p4 = (x - dx_w + dx_l, y - dy_w + dy_l)
            
            poly = Polygon([p1, p2, p3, p4])
            polygons.append(poly)
            areas.append(w * l)
            
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            poly_i = polygons[i]
            area_i = areas[i]
            inds_to_keep = []
            
            for j_idx in range(1, len(order)):
                j = order[j_idx]
                poly_j = polygons[j]
                try:
                    inter_area = poly_i.intersection(poly_j).area
                except:
                    inter_area = 0.0
                    
                union_area = area_i + areas[j] - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                
                if iou <= iou_threshold:
                    inds_to_keep.append(j_idx)
                    
            order = order[inds_to_keep]
            
        return keep

    # =================================================================================
    # LÓGICA DE PROCESAMIENTO PRINCIPAL
    # =================================================================================

    def process_task(self, task_tensors, task_cfg, task_id):
        # 1. Activaciones
        batch_heatmap = self._sigmoid(task_tensors['heatmap'][0])
        batch_reg = task_tensors['reg'][0]
        batch_hei = task_tensors['height'][0]
        batch_dim = np.exp(task_tensors['dim'][0])
        batch_rot = task_tensors['rot'][0]
        
        batch_vel = task_tensors.get('vel', [None])[0]
        batch_iou = self._sigmoid(task_tensors.get('iou', [None])[0]) if 'iou' in task_tensors else None

        # 2. Decode
        boxes3d, raw_scores, labels, iou_scores = self.bbox_coder_decode(
            batch_heatmap, batch_rot[0:1], batch_rot[1:2], batch_hei, batch_dim, batch_vel, 
            reg=batch_reg, iou_scores=batch_iou
        )

        # 3. MÁSCARA 1
        if self.score_threshold > 0.0:
            mask1 = raw_scores > self.score_threshold
            boxes3d = boxes3d[mask1]
            raw_scores = raw_scores[mask1]
            labels = labels[mask1]
            if iou_scores is not None:
                iou_scores = iou_scores[mask1]

        # 4. MÁSCARA 2
        range_mask1 = (boxes3d[:, :3] >= self.post_center_limit_range[:3]).all(axis=1) & \
                      (boxes3d[:, :3] <= self.post_center_limit_range[3:]).all(axis=1)
        boxes3d = boxes3d[range_mask1]
        raw_scores = raw_scores[range_mask1]
        labels = labels[range_mask1]
        if iou_scores is not None:
            iou_scores = iou_scores[range_mask1]

        if len(boxes3d) == 0:
            return boxes3d, raw_scores, labels

        # 5. Rectificación Score
        if iou_scores is not None:
            rectified_scores = np.power(raw_scores, 1 - self.iou_score_beta) * np.power(iou_scores, self.iou_score_beta)
        else:
            rectified_scores = raw_scores

        # 6. MÁSCARA 3
        if self.score_threshold > 0.0:
            mask3 = rectified_scores >= self.score_threshold
            boxes3d = boxes3d[mask3]
            rectified_scores = rectified_scores[mask3]
            labels = labels[mask3]

        if len(boxes3d) == 0:
            return boxes3d, rectified_scores, labels

        # 7. NMS POR TAREA (ENRUTAMIENTO NUMBA VS NUMPY)
        if self.nms_type == 'circle':
            radius = self.min_radius[task_id]
            if self.use_numba:
                keep_inds = self._circle_nms_numba(boxes3d, rectified_scores, labels, radius)
                keep = np.array(keep_inds, dtype=np.int64)
            else:
                keep = self._circle_nms(boxes3d, rectified_scores, labels, radius)
                
        elif self.nms_type == 'rotate': 
            # rotate exacto solo tiene numpy/shapely
            keep = self._rotate_nms_exact_cpu(boxes3d, rectified_scores, self.nms_thr)
            
        elif self.nms_type == 'rotate_aprox':
            if self.use_numba:
                keep_inds = self._rotate_nms_numpy_numba(boxes3d, rectified_scores, self.nms_thr)
                keep = np.array(keep_inds, dtype=np.int64)
            else:
                keep = self._rotate_nms_numpy(boxes3d, rectified_scores, self.nms_thr)
        else:
            print(f"WARNING: NMS type '{self.nms_type}' no reconocido. Se omite NMS.")
            keep = np.arange(len(boxes3d))

        boxes3d = boxes3d[keep]
        rectified_scores = rectified_scores[keep]
        labels = labels[keep]

        # 8. Recorte a Post-Max Size
        if len(boxes3d) > self.post_max_size:
            boxes3d = boxes3d[:self.post_max_size]
            rectified_scores = rectified_scores[:self.post_max_size]
            labels = labels[:self.post_max_size]

        # 9. MÁSCARA 4
        range_mask2 = (boxes3d[:, :3] >= self.post_center_limit_range[:3]).all(axis=1) & \
                      (boxes3d[:, :3] <= self.post_center_limit_range[3:]).all(axis=1)
        
        boxes3d = boxes3d[range_mask2]
        rectified_scores = rectified_scores[range_mask2]
        labels = labels[range_mask2]
            
        return boxes3d, rectified_scores, labels

    def forward(self, flat_outputs):
        task_results = []
        current_idx = 0
        num_headers = len(self.headers)
        
        for task_id, task_cfg in enumerate(self.tasks):
            t_list = flat_outputs[current_idx : current_idx + num_headers]
            current_idx += num_headers
            t_dict = {h: t_list[i] for i, h in enumerate(self.headers) if i < len(t_list)}
            
            boxes, scores, labels = self.process_task(t_dict, task_cfg, task_id)
            
            final_labels = []
            for l in labels:
                if l < len(task_cfg['class_names']):
                    final_labels.append(task_cfg['class_names'][int(l)])
                else:
                    final_labels.append('unknown')
            
            task_results.append((boxes, scores, final_labels))

        all_boxes = []
        all_scores = []
        all_labels = []
        
        for boxes, scores, labels in task_results:
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.extend(labels)
                
        if not all_boxes: return []
            
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)

        all_boxes[:, 2] = all_boxes[:, 2] - all_boxes[:, 5] * 0.5
        
        final_output = []
        for i in range(len(all_boxes)):
            box = all_boxes[i]
            res = {
                'box': {
                    'x': float(box[0]), 'y': float(box[1]), 'z': float(box[2]),
                    'l': float(box[3]), 'w': float(box[4]), 'h': float(box[5]),
                    'rot': float(box[6])
                },
                'velocity': [float(box[7]), float(box[8])] if box.shape[0] > 7 else [0.0, 0.0],
                'score': float(all_scores[i]),
                'label': all_labels[i]
            }
            final_output.append(res)
            
        return final_output
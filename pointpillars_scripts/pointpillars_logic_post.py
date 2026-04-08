import numpy as np
import numba

# =================================================================================
# KERNELS DE NMS (OPTIMIZADOS CON NUMBA)
# =================================================================================

@numba.jit(nopython=True)
def _nms_rotate_aprox_kernel(boxes, scores, threshold):
    """NMS Rotado rápido usando AABB (Axis Aligned Bounding Boxes)."""
    ndets = boxes.shape[0]
    if ndets == 0: return np.zeros(0, dtype=np.int32)

    x, y, w, l, rot = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    
    cos_rot = np.abs(np.cos(rot))
    sin_rot = np.abs(np.sin(rot))
    w_aabb = w * cos_rot + l * sin_rot
    l_aabb = w * sin_rot + l * cos_rot
    
    x1, y1 = x - w_aabb / 2.0, y - l_aabb / 2.0
    x2, y2 = x + w_aabb / 2.0, y + l_aabb / 2.0
    areas = w_aabb * l_aabb

    order = scores.argsort()[::-1].astype(np.int32)
    suppressed = np.zeros(ndets, dtype=np.int32)
    keep = np.zeros(ndets, dtype=np.int32)
    count = 0

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1: continue
        keep[count] = i
        count += 1
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1: continue
            
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            
            w_inter = max(0.0, xx2 - xx1)
            h_inter = max(0.0, yy2 - yy1)
            inter = w_inter * h_inter
            iou = inter / (areas[i] + areas[j] - inter)
            
            if iou > threshold:
                suppressed[j] = 1
    return keep[:count]

def _nms_rotate_exact_cpu(boxes, scores, threshold):
    """NMS Rotado exacto usando Shapely (Lento pero preciso)."""
    from shapely.geometry import Polygon
    if len(boxes) == 0: return np.zeros(0, dtype=np.int32)

    polygons, areas = [], []
    for box in boxes:
        x, y, w, l, rot = box[0], box[1], box[2], box[3], box[4]
        cos_a, sin_a = np.cos(rot), np.sin(rot)
        dx_w, dy_w = (w / 2) * cos_a, (w / 2) * sin_a
        dx_l, dy_l = (l / 2) * -sin_a, (l / 2) * cos_a
        p = [(x+dx_w+dx_l, y+dy_w+dy_l), (x+dx_w-dx_l, y+dy_w-dy_l), 
             (x-dx_w-dx_l, y-dy_w-dy_l), (x-dx_w+dx_l, y-dy_w+dy_l)]
        polygons.append(Polygon(p)); areas.append(w * l)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        poly_i, area_i = polygons[i], areas[i]
        inds_to_keep = []
        for j_idx in range(1, len(order)):
            j = order[j_idx]
            try: inter = poly_i.intersection(polygons[j]).area
            except: inter = 0.0
            if inter / (area_i + areas[j] - inter) <= threshold:
                inds_to_keep.append(j_idx)
        order = order[inds_to_keep]
    return np.array(keep, dtype=np.int32)

# =====================================================================
# CLASE PRINCIPAL
# =====================================================================

class PointPillarsPostProcessor:
    def __init__(self, config):
        self.num_classes = config.num_classes
        self.class_names = config.class_names
        self.dir_offset = config.dir_offset
        self.nms_pre = config.nms_pre
        self.score_thr = config.score_thr
        self.nms_thr = config.nms_thr
        self.max_num = config.max_num
        self.nms_type = getattr(config, 'nms_type', 'rotate_aprox')
        print(f"NMS TYPE:", self.nms_type)
        self.dir_limit_offset = config.dir_limit_offset

        self.anchor_ranges = np.array(config.anchor_generator['ranges'], dtype=np.float32)
        self.anchor_sizes = np.array(config.anchor_generator['sizes'], dtype=np.float32)
        self.anchor_rotations = np.array(config.anchor_generator['rotations'], dtype=np.float32)
        
        self.anchors = None
        self.last_feat_shape = None

    def _generate_anchors(self, featmap_size):
        H, W = featmap_size 
        num_sizes, num_rots = len(self.anchor_sizes), len(self.anchor_rotations)
        anchors = np.zeros((H, W, num_sizes, num_rots, 7), dtype=np.float32)
        
        for i in range(num_sizes):
            r = self.anchor_ranges[i]
            x_c = np.linspace(r[0], r[3], W + 1, dtype=np.float32)
            y_c = np.linspace(r[1], r[4], H + 1, dtype=np.float32)
            x_s, y_s = (x_c[1]-x_c[0])/2, (y_c[1]-y_c[0])/2
            YY, XX = np.meshgrid(y_c[:H] + y_s, x_c[:W] + x_s, indexing='ij')
            
            for r_idx, rot in enumerate(self.anchor_rotations):
                anchors[:, :, i, r_idx, 0:2] = np.stack([XX, YY], axis=-1)
                anchors[:, :, i, r_idx, 2] = r[2]
                anchors[:, :, i, r_idx, 3:6] = self.anchor_sizes[i]
                anchors[:, :, i, r_idx, 6] = rot
        return anchors.reshape(1, -1, 7)

    def _decode_boxes(self, anchors, deltas):
        xa, ya, za, wa, la, ha, ra = [anchors[:, i] for i in range(7)]
        xt, yt, zt, wt, lt, ht, rt = [deltas[:, i] for i in range(7)]
        
        za_off = za + ha / 2
        diag = np.sqrt(la**2 + wa**2)
        
        xg, yg, zg = xt * diag + xa, yt * diag + ya, zt * ha + za_off
        lg, wg, hg = np.exp(lt) * la, np.exp(wt) * wa, np.exp(ht) * ha
        rg = rt + ra
        
        vx = deltas[:, 7] if deltas.shape[1] > 7 else np.zeros_like(xt)
        vy = deltas[:, 8] if deltas.shape[1] > 8 else np.zeros_like(xt)
        
        return np.stack([xg, yg, zg - hg/2, wg, lg, hg, rg, vx, vy], axis=-1)

    def _topk_and_filter(self, cls_scores, bbox_preds, dir_cls_preds):
        _, C, H, W = cls_scores.shape
        if self.anchors is None or self.last_feat_shape != (H, W):
            self.anchors = self._generate_anchors((H, W))
            self.last_feat_shape = (H, W)

        scores_raw = cls_scores[0].transpose(1, 2, 0).reshape(-1, self.num_classes)
        deltas_raw = bbox_preds[0].transpose(1, 2, 0).reshape(-1, bbox_preds.shape[1] // (C // self.num_classes))
        dir_raw = dir_cls_preds[0].transpose(1, 2, 0).reshape(-1, 2)
        
        scores = 1 / (1 + np.exp(-np.clip(scores_raw, -50, 50)))
        max_scores, labels = np.max(scores, axis=1), np.argmax(scores, axis=1)
        
        valid_mask = max_scores > self.score_thr
        if not np.any(valid_mask): return [np.array([])]*5

        if np.sum(valid_mask) > self.nms_pre:
            cand_indices = np.where(valid_mask)[0]
            topk_local = np.argpartition(max_scores[valid_mask], -self.nms_pre)[-self.nms_pre:]
            final_indices = cand_indices[topk_local]
            valid_mask = np.zeros_like(max_scores, dtype=bool)
            valid_mask[final_indices] = True

        return max_scores[valid_mask], labels[valid_mask], deltas_raw[valid_mask], \
               dir_raw[valid_mask], self.anchors[0][valid_mask]

    def _multiclass_nms(self, boxes, scores, labels):
        all_keep = []
        for cls_id in range(self.num_classes):
            mask = labels == cls_id
            if not np.any(mask): continue
            
            cls_boxes, cls_scores = boxes[mask], scores[mask]
            nms_boxes = cls_boxes[:, [0, 1, 3, 4, 6]]
            
            if self.nms_type == 'rotate_exact':
                keep = _nms_rotate_exact_cpu(nms_boxes, cls_scores, self.nms_thr)
            else:
                keep = _nms_rotate_aprox_kernel(nms_boxes, cls_scores, self.nms_thr)
                
            all_keep.append(np.where(mask)[0][keep])
            
        return np.concatenate(all_keep) if all_keep else np.array([], dtype=np.int32)

    def forward(self, outs):
        # 1. Filtro Top-K y Confidence
        res = self._topk_and_filter(outs[0], outs[1], outs[2])
        scores, labels, deltas, dir_p, v_anchors = res
        if len(scores) == 0: return []
        
        # 2. Decoding
        boxes = self._decode_boxes(v_anchors, deltas)
        
        # 3. Dirección (Lógica limit_period corregida)
        dir_labels = np.argmax(dir_p, axis=-1)
        
        # Parámetros del periodo (Lógica mmdet3d oficial)
        period = np.pi
        val = boxes[:, 6] - self.dir_offset
        
        # Aplicamos la fórmula limit_period: val - floor(val / period + offset) * period
        dir_rot = val - np.floor(val / period + self.dir_limit_offset) * period
        
        # Re-construimos el ángulo final: sumamos el offset original y el salto de 180º (pi)
        boxes[:, 6] = dir_rot + self.dir_offset + period * dir_labels
        
        # 4. NMS MULTICLASE
        keep_idx = self._multiclass_nms(boxes, scores, labels)
        
        f_boxes = boxes[keep_idx]
        f_scores = scores[keep_idx]
        f_labels = labels[keep_idx]
        
        # 5. Límite Global (max_num=500)
        if len(f_boxes) > self.max_num:
            idx = np.argsort(-f_scores)[:self.max_num]
            f_boxes, f_scores, f_labels = f_boxes[idx], f_scores[idx], f_labels[idx]
            
        # 6. Formateo Final
        final_output = []
        for i in range(len(f_boxes)):
            b = f_boxes[i]
            final_output.append({
                'box': {'x': float(b[0]), 'y': float(b[1]), 'z': float(b[2]), 
                        'w': float(b[3]), 'l': float(b[4]), 'h': float(b[5]), 'rot': float(b[6])},
                'velocity': [float(b[7]), float(b[8])],
                'score': float(f_scores[i]),
                'label': self.class_names[f_labels[i]]
            })
        return final_output
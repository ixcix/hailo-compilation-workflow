import numpy as np
import glob
import copy
import pickle
import os

from pcdet.datasets import DatasetTemplate
from pcdet.datasets.kitti import kitti_utils
from pcdet.utils import box_utils, calibration_kitti


class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.custom_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    def include_data(self, mode):
        self.logger.info('Loading Custom dataset.')
        custom_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)
        self.custom_infos.extend(custom_infos)
        self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))

    def get_lidar(self, idx: str):
        """Supports both KITTI-like (bin) and custom (npy) structures."""
        bin_path = self.root_path / 'training' / 'velodyne' / f'{idx}.bin'
        npy_path = self.root_path / 'points' / f'{idx}.npy'
        if bin_path.exists():
            return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        if npy_path.exists():
            return np.load(npy_path)
        raise FileNotFoundError(f'No .bin or .npy found for {idx} in {bin_path} or {npy_path}')

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs
        return len(self.custom_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)

        info = copy.deepcopy(self.custom_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)

        input_dict = {
            'frame_id': self.sample_id_list[index],  # string
            'points': points
        }

        if 'annos' in info:  # optional GT
            annos = common_utils.drop_info_with_name(info['annos'], name='DontCare')
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['gt_boxes_lidar']
            })

        return self.prepare_data(data_dict=input_dict)

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.custom_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
        
            # Transform predictions and GT to KITTI format
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]

        metric = kwargs.get('eval_metric', 'kitti')
        if metric == 'kitti':
            return kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        raise NotImplementedError


class KittiDataset(DatasetTemplate):
    """
    Minimal KITTI dataset for inference:
      - loads sample ids and infos from INFO_PATH in cfg
      - provides lidar points + calib to the model
      - attaches image_shape from infos (for 2D bbox projection)
      - converts model outputs to KITTI-style dicts via generate_prediction_dicts
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)
        self.kitti_infos.extend(kitti_infos)
        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    # ---------- I/O ----------
    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists(), f'Missing lidar file: {lidar_file}'
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Filter points that project inside image bounds and with positive depth.
        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists(), f'Missing calib file: {calib_file}'
        return calibration_kitti.Calibration(calib_file)

    # ---------- Torch Dataset API ----------
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        return len(self.kitti_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']  # (H, W) from infos
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['image_shape'] = img_shape
        return data_dict

    # ---------- Predictions -> KITTI dicts ----------
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Convert model outputs to KITTI-style dictionaries (one per sample).

        Args:
            batch_dict: contains 'frame_id', 'calib', 'image_shape'
            pred_dicts: list of dicts with keys:
                - pred_boxes: (N, 7) LIDAR [x,y,z,dx,dy,dz,heading]
                - pred_scores: (N,)
                - pred_labels: (N,)   (1-based class ids)
            class_names: list of class names
            output_path: optional folder to also dump per-frame .txt in KITTI format
        """
        def get_template_prediction(num_samples):
            return {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            pred = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            if hasattr(image_shape, 'cpu'):
                image_shape = image_shape.cpu().numpy()

            # LiDAR -> camera
            boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                boxes_cam, calib, image_shape=image_shape
            )

            pred['name'] = np.array(class_names)[pred_labels - 1]
            pred['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + boxes_cam[:, 6]
            pred['bbox'] = boxes_img
            pred['dimensions'] = boxes_cam[:, 3:6]      # (l, h, w) in camera coords
            pred['location'] = boxes_cam[:, 0:3]
            pred['rotation_y'] = boxes_cam[:, 6]
            pred['score'] = pred_scores
            pred['boxes_lidar'] = pred_boxes
            return pred

        annos = []
        for i, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][i]
            single = generate_single_sample_dict(i, box_dict)
            single['frame_id'] = frame_id
            annos.append(single)

            if output_path is not None:
                cur_det_file = output_path / f'{frame_id}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = single['bbox']
                    loc = single['location']
                    dims = single['dimensions']  # (l,h,w) -> save as (h,w,l) in KITTI txt
                    for j in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single['name'][j], single['alpha'][j],
                                 bbox[j][0], bbox[j][1], bbox[j][2], bbox[j][3],
                                 dims[j][1], dims[j][2], dims[j][0],  # h, w, l
                                 loc[j][0], loc[j][1], loc[j][2],
                                 single['rotation_y'][j], single['score'][j]), file=f)
        return annos

class DemoDataset(DatasetTemplate):
    """ Copied from OpenPCDet/tools/demo.py - 
       - no change in this case, just to collect all utils in one place
    """
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

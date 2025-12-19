import os
import pickle
import numpy as np
import open3d as o3d
import sys

abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(abs_path)

try:
    import open3d
    from src import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from src import visualize_utils as V
    OPEN3D_FLAG = False

def load_point_cloud(file_path):
    ext = os.path.splitext(file_path)[1]
    
   
    if ext == '.bin':
        point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity
    elif ext == '.npy':
        point_cloud = np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return point_cloud[:, :3]  # Solo x, y, z

def load_labels_from_pickle(pickle_file, dataset):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    labels_dict = {}
    
    for sample in data:
        print(sample.keys())  # Imprime las claves principales
        #print(sample['annos'].keys())  # Imprime las claves dentro de 'annos'
        # if dataset == 'waymo':
        #     frame_id = sample['frame_id'][-3:].zfill(4)
        #     # Accede a las cajas, puntuaciones y etiquetas en 'gt_boxes_lidar'
        #     labels_dict[frame_id] = {
        #         'boxes': sample['annos']['gt_boxes_lidar'],  # Cajas de las anotaciones reales (ground truth)
        #         'labels': sample['annos']['obj_ids'],  # Probablemente las etiquetas de los objetos
        #         # 'scores': sample['annos']['pred_scores']  # Aquí no parece haber puntuaciones, así que quizás se deben agregar de otro lado
        #     }
        # else:
        frame_id = os.path.splitext(sample['sample_name'])[0]  # "000009"
        labels_dict[frame_id] = {
            'boxes': sample['pred_boxes'],
            'scores': sample['pred_scores'],
            'labels': sample['pred_labels']
        }
        
        
    return labels_dict

def create_open3d_bounding_box(box, color=[1, 0, 0]):
    """
    box: [x, y, z, length, width, height, yaw]
    """
    if hasattr(box, 'cpu'):
        box = box.cpu().numpy()
    else:
        box = np.array(box)

    
    if len(box) == 7:
        x, y, z, l, w, h, yaw = box
    elif len(box) == 9:
        x, y, z, l, w, h, yaw = box[:7]


    center = np.array([x, y, z], dtype=np.float64)
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = np.array([l, w, h], dtype=np.float64)
    bbox.R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle(np.array([0, 0, yaw], dtype=np.float64))
    bbox.color = color
    return bbox

def visualize_point_cloud_with_boxes(point_cloud, boxes):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    for box in boxes:
        bbox = create_open3d_bounding_box(box)
        vis.add_geometry(bbox)

    vis.run()
    vis.destroy_window()

    

if __name__ == "__main__":
    dataset = "kitti"  # waymo kitti or innovizone
    pickle_file = f'../output/pointpillars/{dataset}/results.pkl'
    # pickle_file = f'/local/shared_with_docker/PointPillars/data/{dataset}/segment-15832924468527961_1564_160_1584_160_with_camera_labels.pkl'
    pointcloud_dir = f'../data/{dataset}/velodyne_val'
    # pointcloud_dir = f'./data/mis_nubes/'

    # Cargar las etiquetas
    labels_dict = load_labels_from_pickle(pickle_file, dataset)

    for frame_id, label_data in labels_dict.items():
        # Ruta de la nube de puntos
        bin_path = os.path.join(pointcloud_dir, f"{frame_id}.bin")
        npy_path = os.path.join(pointcloud_dir, f"{frame_id}.npy")

        if os.path.exists(bin_path):
            points = load_point_cloud(bin_path)
        elif os.path.exists(npy_path):
            points = load_point_cloud(npy_path)
            print(npy_path)
        else:
            print(f"No point cloud found for {frame_id}")
            continue

        print(label_data.keys())
        boxes = label_data['boxes']
        print(f"Visualizing frame: {frame_id} with {len(boxes)} boxes")
        # visualize_point_cloud_with_boxes(points, boxes)
        
        print("boxes type:", type(label_data))
        print("boxes content:", label_data)
        
        print(len(label_data))
        print(len(label_data['boxes']))
        
        V.draw_scenes(
                points=points, ref_boxes=label_data['boxes'],
                ref_scores=label_data['scores'], ref_labels=label_data['labels']
            )
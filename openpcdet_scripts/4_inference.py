
from multiprocessing import Process, Queue
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
 InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType, HailoStreamDirection)

import os
import sys
import numpy as np
from pathlib import Path
import pickle
import time
import tensorflow as tf
import hailo_sdk_client
print(hailo_sdk_client.__version__)


# Obtener la ruta absoluta de la carpeta "src"
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

#import openpcdet2hailo_utilsCPU as ohu;

import openpcdet2hailo_utils as ohu;


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network




# Replace with your OpenPCDet clone directory
openpcdet_clonedir = '/local/shared_with_docker/PointPillars/src/OpenPCDet'
sys.path.append(openpcdet_clonedir + '/tools/')

################################# PATHS and config #################################
model = 'pointpillars'
dataset = 'kitti' #kitti, waymo or innovizone

# Paths to model configuration and weights
yaml_name = f'/local/shared_with_docker/PointPillars/cfgs/{model}_{dataset}.yaml'
pth_name = f'/local/shared_with_docker/PointPillars/model/{model}_{dataset}.pth'

# Path to point cloud data custom
# sample_pointclouds = f'/local/shared_with_docker/PointPillars/data/{dataset}'
# demo_pointcloud = f'/local/shared_with_docker/PointPillars/data/{dataset}/'
# sample_pointclouds = f'/local/shared_with_docker/PointPillars/data/mis_nubes'
# demo_pointcloud = f'/local/shared_with_docker/PointPillars/data/mis_nubes/'
pointclouds = f'/local/shared_with_docker/PointPillars/data/{dataset}/velodyne_val'

# File extension of point cloud files
if dataset == 'kitti':
    pc_file_extention = '.bin'
else:
    pc_file_extention = '.npy'


# Path to output dir for the onnx file
output_path = f'/local/shared_with_docker/PointPillars/output/{model}/{dataset}'

# Hardware architecture
hw_arch='hailo8'

hef_name = f'{output_path}/pp_bev_w_head.hef'


# Directory where logs should be saved
log_dir = '/local/shared_with_docker/PointPillarsHailoInnoviz/logs'
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
log_file = os.path.join(log_dir, 'pp_to_onnx.log')
logger = common_utils.create_logger(log_file)
logger = common_utils.create_logger()

def get_model(cfg, pth_name, demo_dataset):    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
    model.eval()
    return model

def cfg_from_yaml_file_wrap(yaml_name, cfg):
    cwd = os.getcwd()
    os.chdir(openpcdet_clonedir+'/tools/')
    cfg_from_yaml_file(yaml_name, cfg)
    os.chdir(cwd)


def send_from_queue(configured_network, read_q, num_images, start_time):
    """ Bridging a queue into Hailo platform FEED. To be run as a separate process. 
        Reads (preprocessed) images from a given queue, and sends them serially to Hailo platform.        
    """    
    configured_network.wait_for_activation(1000)
    vstreams_params = InputVStreamParams.make(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    print('Starting sending input images to HW inference...\n')
    with InputVStreams(configured_network, vstreams_params) as vstreams:
        vstream_to_buffer = {vstream: np.ndarray([1] + list(vstream.shape), dtype=vstream.dtype) for vstream in vstreams}
        for i in range(num_images):
            hailo_inp = read_q.get()
            for vstream, _ in vstream_to_buffer.items():                                
                vstream.send(hailo_inp)
            print(f'sent img #{i}')
    #print(F'Finished send after {(time.time()-start_time) :.1f}')
    return 0

def recv_to_queue(configured_network, write_q, num_images, start_time):
    """ Bridging Hailo platform OUTPUT into a queue. To be run as a separate process. 
        Reads output data from Hailo platform and sends them serially to a given queue.
    """
    configured_network.wait_for_activation(1000)
    vstreams_params = OutputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    print('Starting receving HW inference output..\n')
    with OutputVStreams(configured_network, vstreams_params) as vstreams:
        # print('vstreams_params', vstreams_params)
        for i in range(num_images):            
            hailo_out = {vstream.name: np.expand_dims(vstream.recv(), 0) for vstream in vstreams}    
            write_q.put(hailo_out)
            print(f'received img #{i}')
    #print(F'Finished recv after {time.time()-start_time :.1f}')
    return 0

def generate_data_dicts(demo_dataset, num_images, pp_pre_bev_w_head):
    for idx, data_dict in enumerate(demo_dataset):
        if idx >= num_images:
            break
        data_dict = demo_dataset.collate_batch([data_dict])
        ohu.load_data_to_CPU(data_dict)
        # Add sample_name to data_dict with only the file name
        data_dict['sample_name'] = os.path.basename(demo_dataset.sample_file_list[idx])
        # ------ (!) Applying torch PRE-processing -------
        data_dict = pp_pre_bev_w_head.forward(data_dict)
        # ------------------------------------------------
        #logger.info(f'preprocessed sample #{idx}')
        yield data_dict

def generate_hailo_inputs(demo_dataset, num_images, pp_pre_bev_w_head):
    """ generator-style encapsulation for preprocessing inputs for Hailo HW feed
    """
    for data_dict in generate_data_dicts(demo_dataset, num_images, pp_pre_bev_w_head):
        spatial_features = data_dict['spatial_features']
        spatial_features_hailoinp = np.transpose(spatial_features.cpu().detach().numpy(), (0, 2, 3, 1))
        yield data_dict, spatial_features_hailoinp

def post_proc_from_queue(recv_queue, num_images, pp_post_bev_w_head,
                         output_layers_order=['model/concat1', 'model/conv19', 'model/conv18', 'model/conv20']):
    results = []
    with open(f"{output_path}/results.pkl", "wb") as f:
        for i in range(num_images):
            t_ = time.time()
            while(recv_queue.empty() and time.time()-t_ < 3):
                time.sleep(0.01)
            if recv_queue.empty():
                print("RECEIVE TIMEOUT!")
                break
            hailo_out = recv_queue.get(0)
            bev_out = [hailo_out[lname] for lname in output_layers_order]
            
            # ------ (!) Applying torch POST-processing -------
            pred_dicts = pp_post_bev_w_head(bev_out)

            # ------------------------------------------------
            # Add sample_name to each prediction dictionary
            sample_name = recv_queue.sample_names[i]

            # Add 'sample_name' to the dictionary
            pred_dicts['sample_name'] = sample_name
            #print(f'pred_dicts sample name : {sample_name}')
            # Append the dictionary to results
            results.append(pred_dicts)

        # Guardar todos los resultados en un .pkl
        # print(f'RESULTADOS GUARDADOS EN EL PKL: {results}')
        pickle.dump(results, f)
    
    return results



data_source = pointclouds  # replace by a folder for a more serious test
num_images = len(os.listdir(data_source))

cfg_from_yaml_file_wrap(yaml_name, cfg)

demo_dataset = ohu.DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    root_path=Path(data_source), ext=pc_file_extention, logger=logger
)
model = get_model(cfg, pth_name, demo_dataset)

# Library creates the anchors in cuda by default (applying .cuda() in internal implementation)
model.dense_head.anchors = [anc.cpu() for anc in model.dense_head.anchors]

""" (!) Slicing off the torch model all that happens before and after Hailo
"""

pp_pre_bev_w_head = ohu.PP_Pre_Bev_w_Head(model)
pp_post_bev_w_head = ohu.PP_Post_Bev_w_Head(model)


with VDevice() as target:
    print(f'DEVICE: {target}\n')
    print(f'Lista de dispositivos: {target.get_physical_devices()}')
    print(f'Lista de ids: {target.get_physical_devices_ids()}')

    hef = HEF(hef_name)
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    
    # Modificar el tamaño de página de los streams
    # for stream in configure_params.streams_params_by_name.values():
    #     if stream.direction == HailoStreamDirection.INPUT:
    #         stream.desc_page_size = 4096  # Puedes probar también con 8192, 16384, etc.
    #     elif stream.direction == HailoStreamDirection.OUTPUT:
    #         stream.desc_page_size = 4096
    
    
    
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    recv_queue = Queue()
    send_queue = Queue()
    start_time = time.time()
    results = []
    hw_send_process = Process(target=send_from_queue, args=(network_group, send_queue, num_images, start_time))
    hw_recv_process = Process(target=recv_to_queue, args=(network_group, recv_queue, num_images, start_time))

    sample_names = []

    with network_group.activate(network_group_params):
        hw_recv_process.start()
        hw_send_process.start()

        tik1 = time.time()

        for data_dict, hailo_inp in generate_hailo_inputs(demo_dataset, num_images, pp_pre_bev_w_head):
            send_queue.put(hailo_inp)
            sample_names.append(data_dict['sample_name'])
        
        recv_queue.sample_names = sample_names
        
        # Stop timing after processing
        tok1 = time.time()

        tik2 = time.time()
        results = post_proc_from_queue(recv_queue, num_images, pp_post_bev_w_head)

        # Stop timing after processing
        tok2 = time.time()

        hailo_time = tok1 - tik1
        postproc_time = tok2 - tik2
        total_time = tok2 - tik1
        
        hailo_time_per_image = hailo_time / num_images
        postproc_time_per_image = postproc_time / num_images
        total_time_per_image = total_time / num_images
        
        inference_rate_hz = num_images / total_time

        print(f"Total elapsed time: {total_time:.4f} seconds")
        print(f"Average total time per image: {total_time_per_image:.4f} seconds")
        print(f"Average hailo time per image: {hailo_time_per_image:.4f} seconds")
        print(f"Average postprocess time per image: {postproc_time_per_image:.4f} seconds")
        print(f"Inference rate: {inference_rate_hz:.2f} Hz")
        pred_dicts = results[-1] 

    hw_recv_process.join(10)
    hw_send_process.join(10)

    #print(f"PRED_SCORESS: {pred_dicts['pred_scores']}")  
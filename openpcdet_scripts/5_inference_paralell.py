from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)

import os, sys, time, threading, queue
import numpy as np
from pathlib import Path
import pickle
import tensorflow as tf
import hailo_sdk_client
print(hailo_sdk_client.__version__)

# ----------------- Rutas / imports OpenPCDet -----------------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

import openpcdet2hailo_utils as ohu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network

openpcdet_clonedir = '/local/shared_with_docker/PointPillars/src/OpenPCDet'
sys.path.append(openpcdet_clonedir + '/tools/')

# ----------------- Config usuario -----------------
model   = 'pointpillars'
dataset = 'kitti'  # 'kitti' | 'custom' | 'waymo' (TODO)

yaml_name = f'/local/shared_with_docker/PointPillars/cfgs/{model}_{dataset}.yaml'
pth_name  = f'/local/shared_with_docker/PointPillars/model/{model}_{dataset}.pth'

# Para inferencia con datasets reales:
#  - KITTI: raíz con 'training/', 'testing/', 'ImageSets/', etc.
#  - CUSTOM: raíz según tu estructura (ohu.CustomDataset)
data_root = f'/local/shared_with_docker/PointPillars/data/{dataset}'

# Salida y HEF
output_path = f'/local/shared_with_docker/PointPillars/output/{model}/{dataset}'
os.makedirs(output_path, exist_ok=True)
hef_name = f'{output_path}/pp_bev_w_head.hef'  # ajusta si tu .hef está en otro sitio


# Logs
log_dir = '/local/shared_with_docker/PointPillarsHailoInnoviz/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'pp_to_onnx.log')
logger = common_utils.create_logger(log_file)
logger = common_utils.create_logger()


# ----------------- Funciones auxiliares -----------------
def get_model(cfg, pth_name, dataset_inst):
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset_inst)
    model.load_params_from_file(filename=pth_name, logger=logger, to_cpu=True)
    model.eval()
    return model

def cfg_from_yaml_file_wrap(yaml_name, cfg):
    cwd = os.getcwd()
    os.chdir(openpcdet_clonedir+'/tools/')
    cfg_from_yaml_file(yaml_name, cfg)
    os.chdir(cwd)

def generate_data_dicts(dataset_inst, num_images, pp_pre_bev_w_head):
    """
    Itera el dataset, colatea, pasa por el preproc (hasta BEV+head)
    y devuelve (data_dict_batcheado) para que el productor prepare el input Hailo.
    """
    for idx in range(len(dataset_inst)):
        if idx >= num_images:
            break
        data_single = dataset_inst[idx]
        data_dict = dataset_inst.collate_batch([data_single])
        ohu.load_data_to_CPU(data_dict)
        # PRE (torch) hasta BEV+head
        data_dict = pp_pre_bev_w_head.forward(data_dict)
        yield data_dict

def generate_hailo_inputs(dataset_inst, num_images, pp_pre_bev_w_head):
    """
    Convierte 'spatial_features' a NHWC float32 contiguo para Hailo y empaqueta meta.
    """
    for data_dict in generate_data_dicts(dataset_inst, num_images, pp_pre_bev_w_head):
        spatial_features = data_dict['spatial_features']  # tensor NCHW
        hailo_inp = np.transpose(spatial_features.cpu().detach().numpy(), (0, 2, 3, 1))  # NHWC
        hailo_inp = np.ascontiguousarray(hailo_inp, dtype=np.float32)

        # Meta mínima para formatear KITTI después
        # batch size = 1 en este flujo
        frame_id    = data_dict['frame_id'][0]
        calib       = data_dict['calib'][0]
        image_shape = data_dict['image_shape'][0]
        meta = {'frame_id': frame_id, 'calib': calib, 'image_shape': image_shape}
        yield meta, hailo_inp

# (Versiones multiproceso originales no usadas)
def send_from_queue(*args, **kwargs): pass
def recv_to_queue(*args, **kwargs): pass
def post_proc_from_queue(*args, **kwargs): pass

# ----------------- MAIN -----------------
if __name__ == "__main__":

    # 0) Dataset / modelo
    cfg_from_yaml_file_wrap(yaml_name, cfg)

    global results_name  
    results_name = f'results_{cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_TYPE}_{cfg.MODEL.POST_PROCESSING.SCORE_THRESH}'

    # Instancia dataset según variable 'dataset'
    if dataset == 'kitti':
        dataset_inst = ohu.KittiDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
            training=False, root_path=Path(data_root), logger=logger
        )
    elif dataset == 'custom':
        # Asegúrate de tener ohu.CustomDataset implementado similar a KittiDataset
        dataset_inst = ohu.CustomDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
            training=False, root_path=Path(data_root), logger=logger
        )
    elif dataset == 'waymo':
        raise NotImplementedError("Waymo aún no implementado en este script.")
    else:
        raise ValueError(f"Dataset '{dataset}' no soportado.")

    # Tamaño (frames)
    num_images = len(dataset_inst)
    logger.info(f'Total frames: {num_images}')

    # Modelo OpenPCDet
    model = get_model(cfg, pth_name, dataset_inst)

    # Evita CUDA en anchors para portabilidad
    if hasattr(model, 'dense_head') and hasattr(model.dense_head, 'anchors'):
        model.dense_head.anchors = [anc.cpu() for anc in model.dense_head.anchors]

    # Wrappers pre/post (tu código)
    pp_pre_bev_w_head  = ohu.PP_Pre_Bev_w_Head(model)
    pp_post_bev_w_head = ohu.PP_Post_Bev_w_Head(model)

    # 1) Hailo setup
    with VDevice() as target:
        print(f'DEVICE: {target}\n')
        print(f'Dispositivos físicos: {target.get_physical_devices()}')
        print(f'IDs: {target.get_physical_devices_ids()}')

        hef = HEF(hef_name)
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        in_params  = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
        out_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

        # 2) Colas acotadas
        send_q = queue.Queue(maxsize=4)   # preproc -> hailo
        recv_q = queue.Queue(maxsize=8)   # hailo  -> postproc

        # 3) Métricas
        lock = threading.Lock()
        times = {'preproc': 0.0, 'preproc_cnt': 0,
                 'hailo':   0.0, 'hailo_cnt':   0,
                 'post':    0.0, 'post_cnt':    0}

        def pick_first_or_warn(vstreams_iterable, kind="INPUT"):
            lst = list(vstreams_iterable)
            if not lst:
                raise RuntimeError(f"No hay vstreams de {kind} en el HEF.")
            if len(lst) > 1:
                print(f"[WARN] Hay {len(lst)} vstreams de {kind}; usaré el primero: {lst[0].name}")
            return lst[0], lst

        # 4) Threads

        # Producer: PRE → encola ({meta}, hailo_inp)
        def producer():
            try:
                for meta, hailo_inp in generate_hailo_inputs(dataset_inst, num_images, pp_pre_bev_w_head):
                    t0 = time.time()
                    send_q.put((meta, hailo_inp))  # bloquea si lleno
                    dt = time.time() - t0
                    with lock:
                        times['preproc'] += dt
                        times['preproc_cnt'] += 1
                send_q.put(None)
            except Exception as e:
                print("Producer error:", e)
                send_q.put(None)

        # Hailo worker: send -> recv -> encola ({meta}, hailo_out)
        def hailo_worker():
            try:
                with network_group.activate(network_group_params):
                    with InputVStreams(network_group, in_params) as inputs, \
                         OutputVStreams(network_group, out_params) as outputs:

                        iv, input_list = pick_first_or_warn(inputs, kind="INPUT")
                        out_streams = {vs.name: vs for vs in outputs}
                        if not out_streams:
                            raise RuntimeError("No hay vstreams de OUTPUT en el HEF.")
                        out_names = list(out_streams.keys())
                        # print(f"[INFO] Output vstreams: {out_names}")

                        frame_idx = 0
                        while True:
                            item = send_q.get()
                            if item is None:
                                iv.flush()
                                print("[HAILO] flush() enviado; fin de entradas.")
                                break

                            meta, hailo_inp = item
                            t0 = time.time()

                            iv.send(hailo_inp)
                            hailo_out = {}
                            for name, vs in out_streams.items():
                                out_arr = vs.recv()
                                hailo_out[name] = np.expand_dims(out_arr, 0)

                            dt = time.time() - t0
                            with lock:
                                times['hailo'] += dt
                                times['hailo_cnt'] += 1

                            recv_q.put((meta, hailo_out))
                            frame_idx += 1

            except Exception as e:
                print("Hailo worker error:", e)
            finally:
                recv_q.put(None)

        # Consumer: POST → convierte a formato KITTI y guarda results.pkl
        def postproc_consumer():
            results_kitti = []
            try:
                while True:
                    item = recv_q.get()
                    if item is None:
                        break
                    meta, hailo_out = item
                    t0 = time.time()

                    # Orden de tensores de salida (ajusta a tu .hef)
                    out_order = ['model/concat1', 'model/conv19', 'model/conv18', 'model/conv20']
                    bev_out = [hailo_out[l] for l in out_order]

                    # Post-proc a pred dict (OpenPCDet style)
                    pred_dict = pp_post_bev_w_head(bev_out)  # dict con pred_boxes, pred_scores, pred_labels

                    # Empaquetar un batch_dict mínimo para formatear
                    batch_dict = {
                        'frame_id':    [meta['frame_id']],
                        'calib':       [meta['calib']],
                        'image_shape': [meta['image_shape']]
                    }
                    # generate_prediction_dicts espera lista de pred_dicts
                    annos = dataset_inst.generate_prediction_dicts(
                        batch_dict, [pred_dict], dataset_inst.class_names, output_path=None
                    )
                    # annos es lista de 1 elemento
                    results_kitti.extend(annos)

                    dt = time.time() - t0
                    with lock:
                        times['post'] += dt
                        times['post_cnt'] += 1

                # Save time metrics
                output_dir = Path(output_path, results_name)
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                # dump al final
                with open(f"{output_path}/{results_name}/{results_name}.pkl", "wb") as f:
                    pickle.dump(results_kitti, f)
                print(f"[OK] Guardado {len(results_kitti)} anotaciones en {output_path}/{results_name}/{results_name}.pkl")

            except Exception as e:
                print("Postproc error:", e)

        # 5) Lanzar y medir
        tik = time.time()
        t_prod  = threading.Thread(target=producer, daemon=True)
        t_hailo = threading.Thread(target=hailo_worker, daemon=True)
        t_post  = threading.Thread(target=postproc_consumer, daemon=True)
        t_prod.start(); t_hailo.start(); t_post.start()
        t_prod.join(); t_hailo.join(); t_post.join()
        tok = time.time()

        # 6) Métricas finales
        total = tok - tik
        img_done = times['hailo_cnt']
        fps = (img_done / total) if total > 0 else 0.0
        def avg(t, c): return (t / c) if c else 0.0

        print("\n==== Métricas ====")
        print(f"Imágenes procesadas: {img_done}/{num_images}")
        print(f"Tiempo total: {total:.3f}s  |  Throughput: {fps:.2f} Hz")
        print(f"Preproc:  total {times['preproc']:.3f}s  avg {avg(times['preproc'], times['preproc_cnt']):.4f}s/img")
        print(f"Hailo:    total {times['hailo']:.3f}s    avg {avg(times['hailo'],   times['hailo_cnt']):.4f}s/img")
        print(f"Postproc: total {times['post']:.3f}s     avg {avg(times['post'],    times['post_cnt']):.4f}s/img")
        print("=============\n")

        # Save time metrics
        output_dir = Path(output_path, results_name)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        time_path = output_dir / 'time_metrics.txt'
        results_path = output_dir / f'{results_name}.pkl'

        # 1) Guardar tiempos
        with open(time_path, "w") as f:
            f.write(f"RESULTS PKL PATH: {results_path}\n\n")

            if "POST_PROCESSING" in cfg.MODEL:
                f.write("===== Config POST_PROCESSING =====\n")
                post_cfg = cfg.MODEL.POST_PROCESSING
                for key, val in post_cfg.items():
                    f.write(f"{key}: {val}\n")
                f.write("\n")
            
            f.write("===== Tiempos de inferencia =====\n")
            f.write(f"Imágenes procesadas: {img_done}/{num_images}\n")
            f.write(f"Tiempo total: {total:.3f}s  |  Throughput: {fps:.2f} Hz\n")
            f.write(f"Preproc:  total {times['preproc']:.3f}s  avg {avg(times['preproc'], times['preproc_cnt']):.4f}s/img\n")
            f.write(f"Hailo:    total {times['hailo']:.3f}s    avg {avg(times['hailo'],   times['hailo_cnt']):.4f}s/img\n")
            f.write(f"Postproc: total {times['post']:.3f}s     avg {avg(times['post'],    times['post_cnt']):.4f}s/img\n")
            f.write("=============\n")

            logger.info(f'Saved time metrics to: {time_path}')

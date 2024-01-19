import os
import numpy as np
import json
import argparse
from easydict import EasyDict
import yaml
import warnings
from tracker.model import AB3DMOT, initialize
import time
import sys

# src_path = "/media/taole/ssd1/letaotao/OpenPCDet_New/data/once/data/000080/000080.json"
# annos = json.load(open(src_path, 'r'))
# print("hello")

def get_frame_det(dets_all, frame):

    ori_array = dets_all[dets_all[:, 0] == frame, 0:2] # frame_id class_id
    # other_array = dets_all[dets_all[:, 0] == frame, -1].reshape((-1, 1)) # box confidence
    # additional_info = np.concatenate((ori_array, other_array), axis=1)
    additional_info = ori_array

    # get 3D box
    dets = dets_all[dets_all[:, 0] == frame, 2:] # [x, y, z, h, l, w, theta, score]
    dets_frame = {'dets': dets, 'info': additional_info}
    return dets_frame

def get_timestring():
    return time.strftime('%Y%m%d_%Hh%Mm%Ss')

def Config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')
    cfg = EasyDict(yaml.safe_load(listfile1))
    setting_show = listfile2.read().splitlines()
    listfile1.close()
    listfile2.close()
    return cfg, setting_show

def load_detection(file):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dets = np.loadtxt(file, delimiter=' ')

    if len(dets.shape) == 1: dets = np.expand_dims(dets, aixs=0)
    if dets.shape[1] == 0:
        return [], False
    else:
        return dets, True 

def parse_args():
    parser = argparse.ArgumentParser(description='3D-Tracking')
    parser.add_argument('--dataset', type=str, default='once', help='determine the dataset')
    parser.add_argument('--seq_data', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/data/once/data/000028', help='determine the seq for tracking')
    parser.add_argument('--config', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/tracking_code/config/once.yaml', help="some config for 3D-Tracking")
    parser.add_argument('--save_root', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/tracking_code/trk', help="some config for 3D-Tracking")

    args = parser.parse_args()
    return args

def main_per_cat(cfg, cat, ID_start, seq_data, save_trk_root, log):

    detection_path = cfg.detections
    seq_dets, flag = load_detection(detection_path)
    if not flag:
        print("please input the right detection results !\n")

    # create folders for saving
    # save_trk_root = os.path.split(os.path.split(detection_path)[0])[0] + '/trk'
    seq_name = os.path.split(seq_data)[-1]

    save_dir  = os.path.join(save_trk_root, "centerpoint_{}".format(cat))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_p = os.path.join(save_dir, "trk_res_000028.txt")

    if os.path.exists(file_p):
        os.remove(file_p)

    track_results_file = open(file_p, 'a+')
    # initialize tracker
    tracker, frame_list = initialize(cfg, seq_data, cat, ID_start, log)
    
    # loop over frame
    min_frame, max_frame = int(0), int(len(frame_list) - 1)

    total_time = 0.0
    for frame in range(min_frame, max_frame + 1):
        # logging
        print_str = 'processing %s: %d/%d \r' % (seq_name, frame, max_frame)
        sys.stdout.write(print_str)
        sys.stdout.flush()

        # tracking by detection
        dets_frame = get_frame_det(seq_dets, frame)
        since = time.time()
        results, affi = tracker.track(dets_frame, frame, seq_name)
        total_time += time.time() - since
        res = results[0]
        for i, res_trk in enumerate(res):
            res_str = str(frame) + " " + str(res_trk[0]) + " " + str(res_trk[1]) + " " + str(res_trk[2]) + " " \
                + str(res_trk[3]) + " " + str(res_trk[4]) + " " + str(res_trk[5]) + " " + \
                str(res_trk[6]) + " " + str(res_trk[7]) + "\n"
            track_results_file.write(res_str)
    print("tracking well done !")    
        
def main(args):

    # load config files
    config_path = args.config
    cfg, settings_show = Config(config_path)

    seq_data = args.seq_data

    save_trk_root = args.save_root
    if not os.path.exists(save_trk_root):
        os.makedirs(save_trk_root)

    # print configs
    time_str = get_timestring()

    log_dir = os.path.join(save_trk_root, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_p = os.path.join(log_dir, 'log_%s_%s.txt' % (time_str, cfg.dataset))
    log = open(log_p, 'w')
    
    ID_start = 1

    for cat in cfg.cat_list:
        ID_start = main_per_cat(cfg, cat, ID_start, seq_data, save_trk_root, log)

    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)



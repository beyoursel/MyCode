import argparse
import glob
from pathlib import Path
import json
import os

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
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

    def decode_box(self, anno):
        center = anno['center']
        size = anno['size']
        rotation = anno['rotation']
        box = [center['x'], center['y'], center['z'], size['x'], size['y'], size['z'], rotation['yaw']]
        return box

    def decode_gt_boxes(self, labels):
        gt_labels = labels['labels']
        gt_boxes = [self.decode_box(anno) for anno in gt_labels]
        gt_names = [anno['type'] for anno in gt_labels]
        return np.stack(np.asarray(gt_boxes)), np.asarray(gt_names)

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
            labels = json.load(open(self.sample_file_list[index].replace("points", "label").replace("npy", "json"), 'r', encoding='utf-8'))
        else:
            raise NotImplementedError
        
        # gt_boxes, gt_names = self.decode_gt_boxes(labels)
        # input_dict = {
        #     'points': points,
        #     'frame_id': index,
        #     'gt_boxes': gt_boxes,
        #     'gt_names': gt_names
        # }

        input_dict = {
            'points': points,
            'frame_id': index,
            # 'gt_boxes': gt_boxes,
            # 'gt_names': gt_names
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/tools/cfgs/once_models/centerpoint.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/data/once/data/000028/lidar_roof',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/tools/centerpoint_ori_once.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--result_path', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/tracking_code/detections/result.txt', help='save the detection results')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    if os.path.exists(args.result_path):
        os.remove(args.result_path)
        results_file = open(args.result_path, 'a+')
    else:
        results_file = open(args.result_path, 'a+')
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy().astype(np.float32)
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy().astype(np.float32)
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy().astype(np.float32)

            pred_box = np.concatenate((pred_labels.reshape(-1, 1), pred_boxes, pred_scores.reshape(-1, 1)), axis=1)

            for i in range(pred_box.shape[0]):
                if str(int(pred_box[i][0])) not in ['1']:
                    continue
                str_box = str(idx) + " " + str(int(pred_box[i][0])) + " " +  str(pred_box[i][1]) \
                + " " +  str(pred_box[i][2]) + " " +  str(pred_box[i][3]) + " " +  str(pred_box[i][4]) \
                + " " +  str(pred_box[i][5]) + " " +  str(pred_box[i][6]) + " " +  str(pred_box[i][7]) \
                + " " +  str(pred_box[i][8]) + "\n"
                results_file.write(str_box)
            # print("processing: %d" % idx)
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
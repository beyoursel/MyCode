import argparse
import glob
from pathlib import Path
import json
import os

# import open3d
# from visual_utils import open3d_vis_utils as V
# OPEN3D_FLAG = True

import mayavi.mlab as mlab
from visual_utils import visualize_utils as V
OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


ONCE_CLS = ['Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist']

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

        par_dir, dir_name = os.path.split(os.path.split(str(root_path))[0])
        overall_labels = json.load(open(os.path.join(par_dir, dir_name,  "%s.json" % dir_name), 'r', encoding='utf-8'))['frames']
        self.gt_labels_list = overall_labels
        data_file_list.sort()
        self.sample_file_list = data_file_list


    def decode_box_rs(self, anno):
        center = anno['center']
        size = anno['size']
        rotation = anno['rotation']
        box = [center['x'], center['y'], center['z'], size['x'], size['y'], size['z'], rotation['yaw']]
        return box

    def decode_gt_boxes_rs(self, labels):
        gt_labels = labels['labels']
        gt_boxes = [self.decode_box(anno) for anno in gt_labels]
        gt_names = [anno['type'] for anno in gt_labels]
        return np.stack(np.asarray(gt_boxes)), np.asarray(gt_names)

    # def decode_box_once(self, anno):
    #     center = anno['center']
    #     size = anno['size']
    #     rotation = anno['rotation']
    #     box = [center['x'], center['y'], center['z'], size['x'], size['y'], size['z'], rotation['yaw']]
    #     return box

    def decode_gt_boxes_once(self, labels):
        gt_names = labels['names']
        gt_boxes = labels['boxes_3d']
        return np.stack(np.asarray(gt_boxes)), np.asarray(gt_names)

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
            labels = self.gt_labels_list[index]['annos'] if 'annos' in self.gt_labels_list[index].keys() else None
            if labels is None:
                return 0
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
            labels = json.load(open(self.sample_file_list[index].replace("points", "label").replace("npy", "json"), 'r', encoding='utf-8'))
        else:
            raise NotImplementedError
        
        gt_boxes, gt_names = self.decode_gt_boxes_once(labels)
        input_dict = {
            'raw_points': points,
            'points': points,
            'frame_id': index,
            'gt_boxes': gt_boxes,
            'gt_names': gt_names
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/trained_weights/centerpoint_dyn/centerpoint_dyn.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/data/once/data/000112/lidar_roof',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/trained_weights/centerpoint_dyn/centerpoint_dyn_epoch_72.pt', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--vis_save', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/for_ppt/detection/origi', help='save the viusalized point cloud')
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

    vis_save = args.vis_save
    if not os.path.exists(vis_save):
        os.makedirs(vis_save)  

    # without screen
    # mlab.options.offscreen = True

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            if data_dict == 0:
                continue
            
            # if idx not in [0, 460, 480, 620, 660, 700, 800, 840, 920, 1140, 1560, 1940]:
            #     continue
            # if idx % 20 != 0:
            #     continue
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_labels = pred_dicts[0]['pred_labels']
            pred_boxes = pred_dicts[0]['pred_boxes']
            pred_scores = pred_dicts[0]['pred_scores']
            gt_boxes = data_dict['gt_boxes'][0]
            gt_cls_index = gt_boxes[:, -1].reshape(-1).cpu().numpy().astype(np.int32)
            gt_names = [ONCE_CLS[g_cls-1] for g_cls in gt_cls_index.tolist()]

            V.draw_scenes(
                points=data_dict['raw_points'][0][:, :3], gt_boxes=gt_boxes, ref_boxes=pred_boxes, gt_labels=gt_names,
                ref_scores=pred_scores, ref_labels=pred_labels
            )

            # V.draw_scenes(
            #     points=data_dict['raw_points'][0][:, :3], ref_boxes=pred_boxes, ref_scores=pred_scores, ref_labels=pred_labels
            # )


            # V.draw_scenes(
            #     points=data_dict['raw_points'][0][:, :3]
            # )

            # for once, exclude the ped and cyclist
            # pred_labels = pred_dicts[0]['pred_labels']
            # mask_cls = (pred_labels != 4) * (pred_labels != 5)
            # pred_boxes = pred_dicts[0]['pred_boxes'][mask_cls]
            # pred_scores = pred_dicts[0]['pred_scores'][mask_cls]
            # gt_boxes = data_dict['gt_boxes'][0]
            # mask_gt = (gt_boxes[:, 7] != 4) * (gt_boxes[:, 7] != 5)

            # V.draw_scenes(
            #     points=data_dict['raw_points'][0][:, :3], gt_boxes=gt_boxes[mask_gt], ref_boxes=pred_boxes,
            #     ref_scores=pred_scores # ref_labels=pred_dicts[0]['pred_labels'
            # )

            # V.draw_scenes(
            #     points=data_dict['raw_points'][0][:, :3], gt_boxes=gt_boxes[mask_gt]
            # )

            # save the pictures    
            # f = mlab.gcf()
            # cam = f.scene.camera
            # cam.zoom(2.5) # default=2.5
            # mlab.savefig(filename=vis_save+"/%03d" %(idx+1)+'.bmp') #(保存每帧的点云及框图)
            # mlab.close()

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

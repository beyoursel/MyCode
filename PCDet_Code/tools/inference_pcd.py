import argparse
import glob
from pathlib import Path
import open3d
# try:
# import open3d
# from visual_utils import open3d_vis_utils as V
# OPEN3D_FLAG = True
# except:
import mayavi.mlab as mlab
from visual_utils import visualize_utils as V
OPEN3D_FLAG = False

import numpy as np
import torch
import random
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import os

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.pcd'):
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
        # print("the index is : %d" % index)
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            print(self.sample_file_list[index])
            raw_points = np.asarray(open3d.io.read_point_cloud(str(self.sample_file_list[index])).points)
            mask = np.isfinite(raw_points[:, 0]) & np.isfinite(raw_points[:, 1]) & np.isfinite(raw_points[:, 2])
            points = raw_points[mask, :] # filter the nan
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
            'raw_points': points
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/trained_weights/centerpoint_rs/centerpoint_rs.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/data/robosense_datasets/datasets/new_data/points',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/media/taole/ssd1/letaotao/OpenPCDet_New/trained_weights/centerpoint_rs/centerpoint_rs_voxel_0.15x0.15x0.2.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')

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

    vis_save = "/media/taole/ssd1/letaotao/OpenPCDet_New/vis_dir_paper"
    if not os.path.exists(vis_save):
        os.makedirs(vis_save)

    # without screen
    mlab.options.offscreen = True    
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):

            if (idx % 20 != 0):
                continue
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                    points=data_dict['raw_points'][0][:, :], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                ) # data_dict['points'][:, 1:]


            # save the pictures    
            f = mlab.gcf()
            cam = f.scene.camera
            cam.zoom(2.5) # default=2.5
            mlab.savefig(filename=vis_save+"/%03d" %(idx+1)+'.bmp') #(保存每帧的点云及框图)
            mlab.close()
            print("saved %d" % idx)
            # mlab.show(stop=True)

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')

if __name__ == '__main__':
    set_random_seed(666, deterministic=True)
    main()

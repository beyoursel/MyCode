import glob
import json
import open3d
# from visual_utils import open3d_vis_utils as V

import sys
sys.path.append("/media/taole/ssd1/letaotao/OpenPCDet_New/tools")

import mayavi.mlab as mlab
from visual_utils import visualize_utils as V

import numpy as np
import pickle
import os


def main():

    vis_save = "/media/taole/ssd1/letaotao/OpenPCDet_New/visual_for_datasets"
    dataset = "nuscenes"
    vis_save_dataset = vis_save + "/" + dataset
    if not os.path.exists(vis_save_dataset):
        os.makedirs(vis_save_dataset)

    src_gt = "/media/taole/ssd1/letaotao/OpenPCDet_New/data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_train.pkl"
    annos = pickle.load(open(src_gt, "rb"))

    # without screen
    mlab.options.offscreen = True
    for i, anno in enumerate(annos):

        if (i % 4 != 0):
            continue
        src_pc = anno['lidar_path']
        pc_name = os.path.split(src_pc)[-1].split('.')[0]
        gt_boxes = anno['gt_boxes']
        gt_names = anno['gt_names']
        src_root = "/media/taole/ssd1/letaotao/OpenPCDet_New/data/nuscenes/v1.0-mini"
        src_pc = os.path.join(src_root, src_pc)
        # src_pc = "/media/taole/ssd1/letaotao/OpenPCDet_New/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"
        # points = np.fromfile(str(src_pc), dtype=np.float32).reshape([-1, 5])[:, :4] # count=-1
        print(src_pc)
        points = np.fromfile(src_pc, dtype=np.float32).reshape([-1, 5])[:, :4] # 注意需要指明np.float32，不然点云解析错误
        V.draw_scenes(points=points[:, :3], gt_boxes=gt_boxes[:, 0:7], gt_labels=gt_names)
    
        # save the pictures    
        f = mlab.gcf() # 获得当前的场景
        cam = f.scene.camera
        cam.zoom(2.5) # default=2.5
        mlab.savefig(filename=vis_save_dataset + "/%s" %(pc_name) + '.bmp') #(保存每帧的点云及框图)
        mlab.close()

        # mlab.show(stop=True)
        print("well-done !")

if __name__ == "__main__":
    main()
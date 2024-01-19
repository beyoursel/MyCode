import glob
import json
import open3d
import sys

sys.path.append("/media/taole/ssd1/letaotao/OpenPCDet_New/tools")
from visual_utils import open3d_vis_utils as V

import mayavi.mlab as mlab


# sys.path.append("/media/taole/ssd1/letaotao/OpenPCDet_New/tools")
# # import visualize_utils as V
# from visual_utils import visualize_utils as V

import numpy as np
import os
import pickle


db_gt_infos = pickle.load(open("/media/taole/ssd1/letaotao/OpenPCDet_New/data/once/once_dbinfos_train.pkl", 'rb'))

src_gt_database = "/media/taole/ssd1/letaotao/OpenPCDet_New/data/once"
vis_save = "/media/taole/ssd1/letaotao/OpenPCDet_New/tools/cls_gt/car"

if not os.path.exists(vis_save):
    os.makedirs(vis_save)

# without screen
# mlab.options.offscreen = True

for i, db_gt in enumerate(db_gt_infos['Car']):
    # if i < 100:
    #     continue
    name = [db_gt['name']]
    pc_path = db_gt['path']
    gt_idx = db_gt['gt_idx']
    box3d_lidar = db_gt['box3d_lidar']

    re_point_p = os.path.join(src_gt_database, pc_path)

    print("visualize: %d" % i)
    re_points = np.fromfile(re_point_p, dtype=np.float32).reshape(-1, 4)

    points = re_points[:, :3] + box3d_lidar[:3]
    # pred_scores = preds[preds[:, 0] == i, -1]
    # pred_labels = preds[preds[:, 0] == i, 1].astype(np.int32)
    # V.draw_scenes(
    #     points=points[:, :3], ref_boxes=det_frame, ref_scores=pred_scores, ref_labels=pred_labels
    # )

    V.draw_scenes(
        points=points[:, :3], gt_boxes=box3d_lidar.reshape(1, -1))
    
    # V.draw_scenes(
    #     points=points[:, :3], gt_boxes=box3d_lidar.reshape(1, -1), gt_labels=name)
    
    # V.draw_scenes(
    #     points=points[:, :3])

    # save the pictures    
    # f = mlab.gcf()
    # cam = f.scene.camera
    # cam.zoom(2.5) # default=2.5
    # mlab.savefig(filename=vis_save+"/%03d" %(i+1)+'.bmp') #(保存每帧的点云及框图)
    # mlab.close()

    # mlab.show(stop=True)
import glob
import json
import open3d
# from visual_utils import open3d_vis_utils as V

import mayavi.mlab as mlab
from visual_utils import visualize_utils as V

import numpy as np
import os

scenes = "000077"

src_bin_path = "/media/taole/ssd1/letaotao/OpenPCDet_New/data/once/data/%s/lidar_roof" % scenes
point_lists = sorted(glob.glob(os.path.join(src_bin_path, "*.bin")))

trks = np.loadtxt("/media/taole/ssd1/letaotao/OpenPCDet_New/tracking_code/trk/centerpoint_Car/trk_res_%s.txt" % scenes, delimiter=' ')
preds = np.loadtxt("/media/taole/ssd1/letaotao/OpenPCDet_New/tracking_code/detections/centerpoint_car_%s.txt" % scenes, delimiter=' ')

vis_save = "/media/taole/ssd1/letaotao/OpenPCDet_New/tracking_code/trk/track_picture"
if not os.path.exists(vis_save):
    os.makedirs(vis_save)

# without screen
# mlab.options.offscreen = True

for i, point_p in enumerate(point_lists):
    # if i < 100:
    #     continue
    print("visualize: %d" % i)
    points = np.fromfile(point_p, dtype=np.float32).reshape(-1, 4)
    trk_frame = trks[trks[:, 0] == i, 1:8]
    trk_id = trks[trks[:, 0] == i, 8]
    det_frame = preds[preds[:, 0] == i, 2:-1]
    # pred_scores = preds[preds[:, 0] == i, -1]
    # pred_labels = preds[preds[:, 0] == i, 1].astype(np.int32)
    # V.draw_scenes(
    #     points=points[:, :3], ref_boxes=det_frame, ref_scores=pred_scores, ref_labels=pred_labels
    # )

    V.draw_scenes(
        points=points[:, :3], gt_boxes=det_frame, ref_boxes=trk_frame, ref_scores=trk_id)
    
    # save the pictures    
    # f = mlab.gcf()
    # cam = f.scene.camera
    # cam.zoom(2.5) # default=2.5
    # mlab.savefig(filename=vis_save+"/%03d" %(i+1)+'.bmp') #(保存每帧的点云及框图)
    # mlab.close()

    mlab.show(stop=True)



import glob
import json
# import open3d
# from visual_utils import open3d_vis_utils as V

import mayavi.mlab as mlab
from visual_utils import visualize_utils as V

import numpy as np
import pickle
import os


def main():


    root_path = "/media/taole/ssd1/letaotao/OpenPCDet_New/data/waymo/waymo_processed_data_v0_5_0"
    data_train = "/media/taole/ssd1/letaotao/OpenPCDet_New/data/waymo/waymo_processed_data_v0_5_0_infos_train.pkl"

    pc_data = pickle.load(open(data_train, "rb"))


    vis_save = "/media/taole/ssd1/letaotao/OpenPCDet_New/visual_for_datasets"
    dataset = "waymo"
    vis_save_dataset = vis_save + "/" + dataset
    if not os.path.exists(vis_save_dataset):
        os.makedirs(vis_save_dataset)

    # without screen
    # mlab.options.offscreen = True
    for i, pc in enumerate(pc_data):

        # if (i % 4 != 0):
        #     continue
        point_cloud = pc['point_cloud']
        lidar_seq = point_cloud['lidar_sequence']

        frame_id = point_cloud['sample_idx']
        annos = pc['annos']

        # load gt anos
        gt_dim = annos['dimensions']
        gt_loc = annos['location']
        gt_orientation = annos['heading_angles']
        gt_names = annos['name']
        gt_boxes = np.concatenate([gt_loc, gt_dim, gt_orientation.reshape(-1, 1)], axis=-1)


        lidar_file = root_path + "/" +  lidar_seq + "/" + ('%04d.npy' % frame_id)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]
        points, NLZ_flag = point_features[:, 0:5], point_features[:, 5]

        V.draw_scenes(points=points[:, :3], gt_boxes=gt_boxes[:, 0:7], gt_labels=gt_names)
    
        # save the pictures    
        # f = mlab.gcf() # 获得当前的场景
        # cam = f.scene.camera
        # cam.zoom(2.5) # default=2.5
        # mlab.savefig(filename=vis_save_dataset + "/%04d" %(i) + '.bmp') #(保存每帧的点云及框图)
        # mlab.close()

        mlab.show(stop=True)
        print("well-done !")

if __name__ == "__main__":
    main()
import glob
import json
import open3d as o3d
from visual_utils import open3d_vis_utils as V

import mayavi.mlab as mlab

import sys
sys.path.append("/media/taole/ssd1/letaotao/OpenPCDet_New/tools")
# import visualize_utils as V
from visual_utils import visualize_utils as V

import numpy as np
import os
import pickle

# without screen
mlab.options.offscreen = True

# pointcloud downsample


pcd = o3d.io.read_point_cloud("/media/taole/ssd1/letaotao/OpenPCDet_New/open3d_code/points/1618648594.929652214.pcd")


# 过滤Nan
points = np.asarray(pcd.points)
mask = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
points_filered = points[mask, :] # filter the nan

pcd.points = o3d.utility.Vector3dVector(points_filered) # convert numpy to pointcloud of Open3d

# 体素下采样
voxel_size = 0.8
downpcd = pcd.voxel_down_sample(voxel_size)
print(downpcd)

# print("->正在可视化下采样点云...")
o3d.visualization.draw_geometries([downpcd])

ori_pc = np.asarray(downpcd.points)
# V.draw_scenes(
#     points=points[:, :3], gt_boxes=box3d_lidar.reshape(1, -1))

# V.draw_scenes(
#     points=points[:, :3], gt_boxes=box3d_lidar.reshape(1, -1), gt_labels=name)

V.draw_scenes(
    points=ori_pc[:, :3])

# save the pictures   

vis_save = "/media/taole/ssd1/letaotao/OpenPCDet_New/down_vis" 
f = mlab.gcf()
cam = f.scene.camera
cam.zoom(2.5) # default=2.5
mlab.savefig(filename=vis_save+"/%03d" % (5) + '.bmp') #(保存每帧的点云及框图)
print("saved !")
mlab.close()

# mlab.show(stop=True)
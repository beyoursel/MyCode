import os
import numpy as np

def read_pcd_binary(pcd_file):

    with open(pcd_file, 'rb') as f:
        # 读取pcd文件头信息
        header = ""
        while True:
            line = f.readline().decode('utf-8')
            header += line
            if line.startswith("DATA"):
                break
        print(header)
        # 从文件内容中读取三维坐标点云数据
        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        data = np.fromfile(f, dtype=dtype)
    return header, data



if __name__ == "__main__":

    src_pc = "/media/taole/ssd1/letaotao/OpenPCDet_New/data/ros_pcd_14/1618648680.330108881.pcd"
    h, d = read_pcd_binary(src_pc)
    print("well done")

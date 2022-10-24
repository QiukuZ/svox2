# 本脚本将scannet数据转换为NSVF的数据格式
# 提供给svox2进行训练与测试

# NSVF格式
# --images
# --images/0_xxxx.png  # 0开始为训练数据
# --images/1_xxxx.png  # 1开始为测试数据
# --pose
# --pose/0_xxxx.txt # 同理如上
# --pose/1_xxxx.txt # txt中为4x4矩阵
# --intrinsics.txt  # 4x4内参矩阵

import os
import numpy as np
import imageio
from tqdm import tqdm

if __name__ == '__main__':
    # train_idex = [830,,855,,870, 913]
    # train_index = [830,837,848,855,864,870,886,904,913,917,926]
    # train_index = [830,833,837,841,845,848,850,855,860,864,870,886,890,904,913,917,920,926]
    test_index = [841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900]
    data_basedir = "/media/qk/GoldData/ScanNet"
    scan_scene = "scene0000_00"
    nvsf_data_save_path = "/home/qk/Documents/NewPipeline/svox2/data/scannet_nvsf_demo1_more_view"

    scene_basedir = os.path.join(data_basedir, "scans", scan_scene)
    # mkdir 
    if not os.path.exists(nvsf_data_save_path):
        os.mkdir(nvsf_data_save_path)

    # Copy image
    if not os.path.exists(os.path.join(nvsf_data_save_path, "images")):
        os.mkdir(os.path.join(nvsf_data_save_path, "images"))
    for i,idx in tqdm(enumerate(train_index)):
        image_path = os.path.join(scene_basedir, "color", f"{idx}.jpg")
        image_save_path = os.path.join(nvsf_data_save_path, "images", "0_" + str(i).zfill(4) + ".png")
        image_color = imageio.imread(image_path)
        imageio.imwrite(image_save_path, image_color)
    for i,idx in tqdm(enumerate(test_index)):
        image_path = os.path.join(scene_basedir, "color", f"{idx}.jpg")
        image_save_path = os.path.join(nvsf_data_save_path, "images", "1_" + str(i).zfill(4) + ".png")
        image_color = imageio.imread(image_path)
        imageio.imwrite(image_save_path, image_color)

    # Copy pose
    if not os.path.exists(os.path.join(nvsf_data_save_path, "poses")):
        os.mkdir(os.path.join(nvsf_data_save_path, "poses"))
    for i,idx in tqdm(enumerate(train_index)):
        pose_path = os.path.join(scene_basedir, "pose", f"{idx}.txt")
        pose_save_path = os.path.join(nvsf_data_save_path, "poses", "0_" + str(i).zfill(4) + ".txt")
        np.savetxt(pose_save_path, np.loadtxt(pose_path))
    for i,idx in tqdm(enumerate(test_index)):
        pose_path = os.path.join(scene_basedir, "pose", f"{idx}.txt")
        pose_save_path = os.path.join(nvsf_data_save_path, "poses", "1_" + str(i).zfill(4) + ".txt")
        np.savetxt(pose_save_path, np.loadtxt(pose_path))

    # Copy intrinsic
    intrinsic_in_path = os.path.join(scene_basedir, "intrinsic/intrinsic_color.txt")
    assert(os.path.exists(intrinsic_in_path))
    intrinsic = np.loadtxt(intrinsic_in_path)
    np.savetxt(os.path.join(nvsf_data_save_path, "intrinsics.txt"), intrinsic)
    print("Done")
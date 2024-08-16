import copy
import pathlib
import pickle

import fire
import numpy as np
from skimage import io as imgio

from pcdet.utils import box_utils
from pcdet.utils.point_cloud import bound_points_jit
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.utils.progress_bar import list_bar as prog_bar

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line.strip()) for line in lines]

def _calculate_num_points_in_gt(data_path, infos, relative_path, remove_outside=True, num_features=4):
    for info in infos:
        if relative_path:
            v_path = str(pathlib.Path(data_path) / info["velodyne_path"])
        else:
            v_path = info["velodyne_path"]
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = info['calib/R0_rect']
        Trv2c = info['calib/Tr_velo_to_cam']
        P2 = info['calib/P2']
        if remove_outside:
            points_v = box_utils.remove_outside_points(points_v, rect, Trv2c, P2, info["img_shape"])

        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes_lidar = box_utils.boxes3d_camera_to_lidar(gt_boxes_camera, rect, Trv2c)
        indices = box_utils.points_in_boxes(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([num_ignored])])
        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)

def create_kitti_info_file(data_path, save_path=None, create_trainval=False, relative_path=True):
    train_img_ids = _read_imageset_file("./data/ImageSets/train.txt")
    val_img_ids = _read_imageset_file("./data/ImageSets/val.txt")
    trainval_img_ids = _read_imageset_file("./data/ImageSets/trainval.txt")
    test_img_ids = _read_imageset_file("./data/ImageSets/test.txt")

    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = pathlib.Path(data_path)
    else:
        save_path = pathlib.Path(save_path)
    kitti_infos_train = KittiDataset.get_image_info(data_path, training=True, velodyne=True, calib=True, image_ids=train_img_ids, relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = save_path / 'kitti_infos_train.pkl'
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    kitti_infos_val = KittiDataset.get_image_info(data_path, training=True, velodyne=True, calib=True, image_ids=val_img_ids, relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / 'kitti_infos_val.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)

    filename = save_path / 'kitti_infos_trainval.pkl'
    print(f"Kitti info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)

    kitti_infos_test = KittiDataset.get_image_info(data_path, training=False, label_info=False, velodyne=True, calib=True, image_ids=test_img_ids, relative_path=relative_path)
    filename = save_path / 'kitti_infos_test.pkl'
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)

# 나머지 함수들도 동일하게 `OpenPCDet` 모듈을 이용하도록 수정 필요

if __name__ == '__main__':
    fire.Fire()

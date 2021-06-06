import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset


class Cub200(Dataset):
    def __init__(self, root, split, img_size, center_crop_scale, label_to_onehot = False):
        super(Cub200, self).__init__()
        if split not in ['train', 'test']:
            raise AttributeError('split should be either "train" or "test".')
        self.root = root
        self.dataset_name = "CUB_200_2011"
        self.dataset_path = os.path.join(self.root, self.dataset_name)
        self.train_test_path = os.path.join(self.dataset_path, "train_test_split.txt")
        self.bounding_boxes_path = os.path.join(self.dataset_path, "bounding_boxes.txt")
        self.image_list_path = os.path.join(self.dataset_path, "images.txt")
        self.label_list_path = os.path.join(self.dataset_path, "image_class_labels.txt")
        self.split = 1 if split == "train" else 0
        self.img_size = img_size
        self.center_crop_scale = center_crop_scale
        self.label_to_onehot = label_to_onehot
        self.num_classes = 200

        self.data_id_list = []
        with open(self.train_test_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                id, sign = line.split(' ')
                if int(sign) == self.split:
                    self.data_id_list.append(int(id))
        
        self.file_list = []
        with open(self.image_list_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                id, file = line.split(' ')
                assert int(id) == len(self.file_list) + 1
                self.file_list.append(file)
        
        self.label_list = []
        with open(self.label_list_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                id, label = line.split(' ')
                assert int(id) == len(self.label_list) + 1
                self.label_list.append(int(label))
        
        self.bounding_boxes = []
        with open(self.bounding_boxes_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                bbox_list = line.split(' ')
                assert len(bbox_list) == 5
                assert int(bbox_list[0]) == len(self.bounding_boxes) + 1
                self.bounding_boxes.append([float(bbox_list[1]), float(bbox_list[2]), float(bbox_list[3]), float(bbox_list[4])])
    
    def __len__(self):
        return len(self.data_id_list)
    
    def __getitem__(self, id):
        id = self.data_id_list[id] - 1
        filename = self.file_list[id]
        bbox = self.bounding_boxes[id]
        label = self.label_list[id]
        file_path = os.path.join(self.dataset_path, 'images', filename)
        img = cv2.imread(file_path, )
        xmin, ymin, xoffset, yoffset = bbox
        center = [int(xmin + xoffset / 2), int(ymin + yoffset / 2)]
        h = max(xoffset, yoffset)
        img = center_crop(
            img,
            output_size = [h, h],
            center = center,
            scale = self.center_crop_scale
        )
        img = cv2.resize(img, (self.img_size, self.img_size))
        if self.label_to_onehot:
            res_label = np.zeros(self.num_classes, dtype = np.float32)
            res_label[label - 1] = 1.0
            label = res_label
        else:
            label = np.array([label], dtype = np.float32)
        img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
        img = np.array(img * 2 - 1).astype(np.float32)
        return img, label


def Cub200Dataset(root, split, img_size, center_crop_scale):
    return Cub200(
        root = root,
        split = split,
        img_size = img_size,
        center_crop_scale = center_crop_scale,
        label_to_onehot = True
    )


# Ref: https://github.com/BassyKuo/CUB200-2011, with few critical modifications
def center_crop(img, output_size, center=None, scale=None):
    """
    Args
    - img: np.ndarray. img.shape=[height,width,3].
    - output_size: tuple or list. output_size=[height, width].
    - center: tuple or list. center=[x,y].
                If cenetr is None, use input image center.
    - scale: float.
                If scale is None, do not scale up the boundary.
    """
    hi, wi = img.shape[:2]
    ho, wo = output_size
    if center:
        x, y = center
        if scale:
            ho *= scale
            wo *= scale
            if ho > hi or wo > wi:
                ho, wo = output_size
        if ho > hi or wo > wi:
            ho = min(hi, wi)
            wo = min(hi, wi)
        bound_left   = int(x - wo / 2)
        bound_right  = int(x + wo / 2)
        bound_top    = int(y - ho / 2)
        bound_bottom = int(y + ho / 2)

        if bound_left < 0:
            offset_w = 0
        elif bound_right > wi:
            offset_w = int(wi - wo)
        else:
            offset_w = int(x - wo / 2)

        if bound_top < 0:
            offset_h = 0
        elif bound_bottom > hi:
            offset_h = int(hi - ho)
        else:
            offset_h = int(y - ho / 2)
    else:
        if scale:
            print ("Scaling deny when center variable is None.")
        try:
            if hi < ho and wi < wo:
                raise ValueError("image is too small. use orginal image.")
        except:
            ho = min(hi, wi)
            wo = ho
        offset_h = int((hi - ho) / 2)
        offset_w = int((wi - wo) / 2)
    return img[offset_h : offset_h + int(ho),
                offset_w : offset_w + int(wo)]

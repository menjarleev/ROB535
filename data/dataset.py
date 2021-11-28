from torch.utils.data import Dataset
from utils.transform import rotate3d, get_bbox
from skimage.transform import resize
import skimage
import os
from glob import glob
import torch
import skimage.io as io
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, base_path, phase='train', holdout=0, k_fold=10, input_size=(256, 512), num_class=0, aug_func=None, augment=False):
        self.phase = phase
        self.input_size = input_size
        self.agu_func = aug_func
        self.augment = augment
        folder = 'test' if phase == 'test' else 'trainval'
        folder_path = os.path.join(base_path, folder)
        if phase =='test':
            files = glob(f'{folder_path}/*/*_image.jpg')
            files.sort()
        else:
            csv = np.loadtxt(f'{folder_path}/trainval_labels.csv', skiprows=1, dtype=str, delimiter=',')
            files = csv[:, 0].tolist()
            files = [f'{folder_path}/{f}_image.jpg' for f in files]
            label = csv[:, 1].astype(np.uint8).tolist()
            self.num_class = num_class
        # k-fold validation setting
        if phase == 'test':
            self.file = files
        elif phase == 'val':
            if holdout == -1:
                holdout = 0
            start = int(len(files) * (holdout / k_fold))
            end = int(len(files) * (holdout + 1) / k_fold)
            self.file = files[start:end]
            self.label = label[start:end]
        else:
            if holdout == -1:
                self.file = files
                self.label = label
            else:
                self.file = files[0:int(len(files) * (holdout / k_fold))]
                self.label = label[0:int(len(files) * (holdout/ k_fold))]
                self.file.extend(files[int(len(files) * (holdout + 1) / k_fold):])
                self.label.extend(label[int(len(files) * (holdout + 1) / k_fold):])

    def __getitem__(self, index):
        file_path = self.file[index]
        img = skimage.img_as_float64(io.imread(file_path))
        if self.phase == 'train':
            if self.augment:
                proj = np.fromfile(file_path.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
                proj.resize([3, 4])
                has_bbox = True
                try:
                    bbox = np.fromfile(file_path.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
                except FileNotFoundError:
                    print('[*] bbox not found.')
                    has_bbox = False
                h, w, _ = img.shape
                (in_h, in_w) = self.input_size
                if has_bbox:
                    bbox = bbox.reshape([-1, 11])
                    b = bbox[0]
                    R = rotate3d(b[0:3])
                    t = b[3:6]
                    sz = b[6:9]
                    vert_3D, edges = get_bbox(-sz / 2, sz / 2)
                    vert_3D = R @ vert_3D + t[:, np.newaxis]

                    vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
                    vert_2D = vert_2D / vert_2D[2, :]
                    min_x = int(np.min(vert_2D[0]))
                    max_x = int(np.max(vert_2D[0]))
                    min_y = int(np.min(vert_2D[1]))
                    max_y = int(np.max(vert_2D[1]))
                    obj_len_x = max(max_x - min_x, in_w)
                    obj_len_y = max(max_y - min_y, in_h)
                    centroid_x = (min_x + max_x) / 2
                    centroid_y = (min_y + max_y) / 2
                    left = max(int(centroid_x - (obj_len_x / 2)), 0)
                    right = min(int(centroid_x + (obj_len_x / 2)) + 1, w - 1)
                    left_idx = np.random.randint(0, left + 1)
                    right_idx = np.random.randint(right, w)
                    top = max(0, int(centroid_y - (obj_len_y / 2)))
                    btm = min(h - 1, int(centroid_y + (obj_len_y / 2)) + 1)
                    top_idx = np.random.randint(0, top + 1)
                    btm_idx = np.random.randint(btm, h)
                    img = img[top_idx:btm_idx, left_idx:right_idx]
                else:
                    free_pixel_x = w - in_w
                    free_pixel_y = h - in_h
                    left_idx = np.random.randint(free_pixel_x)
                    right_idx = min(left_idx + in_w + 1, w)
                    top_idx = np.random.randint(free_pixel_y)
                    btm_idx = min(top_idx + in_h + 1, h)
                    img = img[top_idx: btm_idx, left_idx: right_idx]

        img = resize(img, self.input_size, anti_aliasing=True)
        if self.agu_func is not None and self.phase == 'train':
            img = self.agu_func(img)
        # transpose image
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        # normalize image to range [-0.5, 0.5]
        img = img - 0.5
        item_dict = {}
        item_dict['image'] = img
        if self.phase == 'train' or self.phase == 'val':
            # one hot encoding
            label = np.zeros(self.num_class)
            label[self.label[index]] = 1
            label = torch.from_numpy(label)
            item_dict['label'] = label
        return item_dict

    def __len__(self):
        return len(self.file)

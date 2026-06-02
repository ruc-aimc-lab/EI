# load and process data 
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import random


def uni_spaced_sampling(start, end, num, disturbance=False):
    # similay to np.linspace
    step = (end - start) / num
    indices = [step / 2 + i * step for i in range(num)]
    indices = np.array(indices)
    if disturbance:
        indices = indices + np.random.uniform(-step / 2, step / 2, len(indices))
    
    indices = np.round(indices)
    indices = indices.astype(int)
    return indices


class BaseDataset(Dataset):
    def __init__(self, lstpath, img_root, num_class, mappings, modality_idx, augs, train, only_label):
        self.lstpath = lstpath
        self.img_root = img_root
        self.num_class = num_class  
        self.mappings = mappings 
        self.modality_idx = modality_idx  
        self.augs = augs  
        
        self.train = train  
        self.only_label = only_label  
        
        self.lst = self.read_list(lstpath=lstpath)
        self.format_check()
           
    def read_list(self, lstpath):
        lst = []
        for p in lstpath:
            with open(p) as fin:
                lines = fin.readlines()[:]
                lines = list(map(lambda x: x.strip('\r\n').split('\t'), lines))
                lst += lines
        return lst
    
    def __getitem__(self, index):
        
        line = self.lst[index]
        img_path = line[0]
        img_paths = self.get_img_paths(img_path)  
        """
        {
            modality1: [img_path1, img_path2, ...], 
            modality2: ...
            ...
        }
        """

        if self.only_label:
            imgs = -1
        else:
            imgs = self.get_imgs(img_paths)
        """
        {
            modality1: [img1, img2, ...], 
            modality2: ...
            ...
        }
        """
            
        labels = line[1]
        labels = labels.split(',')
        gt = self.get_gt(labels)
        
        return img_path, imgs, gt

    def get_multi_class_gt(self, labels):
        assert len(labels) == 1
        assert labels[0] in self.mappings, labels
        gt = self.mappings[labels[0]]
        return gt
    
    def get_multi_label_gt(self, labels):
        gt = np.zeros(self.num_class)
        for label in labels:
            if label not in self.mappings:
                continue
            gt[self.mappings[label]] = 1
        return gt
    
    def __len__(self):
        return len(self.lst)
    
    def aug_img(self, img, modality):
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]
        
        img, _ = self.augs[modality].process(img)
        img = np.multiply(img, 1 / 255.0)
        img = (img - image_mean) / image_std
        
        img = np.transpose(img, (2, 0, 1))
        return img
    
    def format_check(self):
        no_exist_imgs = []
        not_used_labels = set({})
        label_count = {}
        for line in self.lst:
            img_path = line[0]
            img_paths = self.get_img_paths(img_path)  
            for idx in img_paths:
                for im_path in img_paths[idx]:
                    if not os.path.exists(im_path):
                        no_exist_imgs.append(im_path)

            labels = line[1]
            labels = labels.split(',')
            for label in labels:
                if label not in self.mappings:
                    not_used_labels = not_used_labels.union({label})
                else:
                    label_count[label] = label_count.get(label, 0) + 1
        print('no_exist_imgs:', no_exist_imgs)   
        print('not_used_labels:', not_used_labels)   
        print('label_count:', label_count)   
        
    def get_img_paths(self, raw_path):
        raise NotImplementedError
    
    def get_imgs(self, img_paths):
        raise NotImplementedError
    
    def get_gt(self, labels):
        raise NotImplementedError
    

class derm7ptDataset(BaseDataset):
    def __init__(self, lstpath, img_root, num_class, mappings, modality_idx, augs, train, only_label):
        super().__init__(lstpath, img_root, num_class, mappings, modality_idx, augs, train, only_label)
        
    def get_img_paths(self, raw_path):
        raw_path = raw_path.split(',')
        
        img_paths = {}
        for idx in self.modality_idx:  # load only specified modality
            img_paths[idx] = [os.path.join(self.img_root, raw_path[idx])]
        return img_paths
    
    def get_imgs(self, img_paths):
        imgs = {}
        for modality in img_paths:
            imgs[modality] = []
            for im_path in img_paths[modality]:
                assert os.path.exists(im_path), im_path
                img = cv2.imread(im_path)[:, :, [2, 1, 0]]
                img = self.aug_img(img, modality)
                imgs[modality].append(img)
            imgs[modality] = np.array(imgs[modality])
        
        return imgs
    
    def get_gt(self, labels):
        return self.get_multi_class_gt(labels)
        

class mmc_amdDataset(BaseDataset):
    def __init__(self, lstpath, img_root, num_class, mappings, modality_idx, augs, train, only_label):
        super().__init__(lstpath, img_root, num_class, mappings, modality_idx, augs, train, only_label)
        
    def get_img_paths(self, raw_path):
        raw_path = raw_path.split(',')
        
        img_paths = {}
        for idx in self.modality_idx:  
            paths = raw_path[idx].split(' ')
            if self.train:
                p = random.sample(paths, 1)[0]
                img_paths[idx] = [os.path.join(self.img_root, p)]
            else:
                p = sorted(paths)[0]
                img_paths[idx] = [os.path.join(self.img_root, p)]
        return img_paths
    
    def get_imgs(self, img_paths):
        imgs = {}
        for modality in img_paths:
            imgs[modality] = []
            for im_path in img_paths[modality]:
                assert os.path.exists(im_path), im_path
                img = cv2.imread(im_path)[:, :, [2, 1, 0]]
                img = self.aug_img(img, modality)
                imgs[modality].append(img)
            imgs[modality] = np.array(imgs[modality])
        
        return imgs
    
    
    def get_gt(self, labels):
        return self.get_multi_class_gt(labels)
        

class mrnetDataset(BaseDataset):
    def __init__(self, lstpath, img_root, num_class, mappings, modality_idx, augs, train, only_label, sampling_num):
        super().__init__(lstpath, img_root, num_class, mappings, modality_idx, augs, train, only_label)
        self.sampling_num = sampling_num
        
    def get_img_paths(self, raw_path):
        raw_path = raw_path.split(',')
        
        img_paths = {}
        for idx in self.modality_idx:  
            img_paths[idx] = [os.path.join(self.img_root, raw_path[idx])]
        return img_paths
    
    def get_imgs(self, img_paths):
        imgs = {}
        for modality in img_paths:
            imgs[modality] = []
            for cube_path in img_paths[modality]:
                assert os.path.exists(cube_path), cube_path
                cube = np.load(cube_path)
                cube_length = cube.shape[0]
                indices = uni_spaced_sampling(0, cube_length - 1, self.sampling_num, disturbance=self.train)
                for i in indices:
                    img = cube[i]   
                    img = np.repeat(img[:, :, np.newaxis], 3, axis=2) 
                    img = self.aug_img(img, modality)
                    imgs[modality].append(img)
            imgs[modality] = np.array(imgs[modality])
        
        return imgs
    
    def get_gt(self, labels):
        return self.get_multi_label_gt(labels)
     
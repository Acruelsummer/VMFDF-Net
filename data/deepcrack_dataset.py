# Author: Yahui Liu <yahui.liu@unitn.it>

import os.path
import random
import cv2
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from data.image_folder import make_dataset
from data.utils import MaskToTensor, get_params, affine_transform
# from models.LIOT import distance_weight_binary_pattern_faster
class DeepCrackDataset(BaseDataset):
    """A dataset class for crack dataset."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # self.img_paths = make_dataset(os.path.join(opt.dataroot, '{}_img'.format(opt.phase)))
        # self.lab_dir = os.path.join(opt.dataroot, '{}_lab'.format(opt.phase))


        self.img_paths = make_dataset(os.path.join(opt.dataroot,opt.dataset,'{}_img'.format(opt.phase)))
        # print(self.img_paths,'666')
        self.lab_dir = os.path.join(opt.dataroot,opt.dataset, '{}_lab'.format(opt.phase))
        # print(self.img_paths,'aaa')
        # 得到图片列表
        # print(self.lab_dir,'bbb')
        # ./datasets/CRACK500/test_lab bbb
        # mean = [126.54589362, 119.86447542, 124.93577637, 119.73361778]
        # std = [107.73784026, 101.97971114, 108.13135899, 102.34676233]
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  # transforms.Normalize(mean=mean, std=std)
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))
                                                                       ])
        self.lab_transform = MaskToTensor()
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            image (tensor) - - an image
            label (tensor) - - its corresponding segmentation
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0]+'.png')
        # print(lab_path,'qqq')
        # ./datasets/CRACK500/test_lab/20160222_080933.png qqq
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        # print(lab,'111222')
        # None 111222
        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        img_h, img_w = img.shape[:2]
        lab_h, lab_w = lab.shape[:2]
        if img_w < img_h:  # If the image is "tall" but it should be "wide"
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            lab = cv2.rotate(lab, cv2.ROTATE_90_CLOCKWISE)

        # adjust the image size
        w, h = self.opt.load_width, self.opt.load_height

        # w,h = 448,448
        if w > 0 or h > 0:
            
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        # img_liot = distance_weight_binary_pattern_faster(img)
        # print("LIOT processed image size:", img_liot.shape)
        # print("Label size:", lab.shape)
        # LIOT processed image size: (256, 256, 4)
        # Label size: (256, 256)
        # binarize segmentation

        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        # 对 lab 图像进行阈值处理，使得图像中像素值大于等于127的部分变为255，其他部分变为0，从而生成一个二值图像。
        # apply flip
        if  self.opt.phase == 'train'and (not self.opt.no_flip) and random.random() > 0.5:
            if random.random() > 0.5:
                img = np.fliplr(img)
                lab = np.fliplr(lab)
            else:
                img = np.flipud(img)
                lab = np.flipud(lab)

        # apply affine transform
        # print(self.opt.use_augment,'if use augment')
        if  self.opt.phase == 'train'and self.opt.use_augment:
            if random.random() > 0.5:
                angle, scale, shift = get_params()
                img = affine_transform(img, angle, scale, shift, w, h)
                lab = affine_transform(lab, angle, scale, shift, w, h)

        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)
        # 处理后的标签只有0和1两种值
        # apply the transform to both A and B
        img = self.img_transforms(Image.fromarray(img))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)

        return {'image': img, 'label': lab, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)

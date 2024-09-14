from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np
import cv2
from segment_anything.utils.transforms import ResizeLongestSide

class Cervix_dataset_four(Dataset):
    def __init__(self, data_root_dir="/data/RawData/Training",
                 train_cls=1,
                 cls=99):
        self.train_cls = train_cls

        self.mask_root_list = sorted(os.listdir(osp.join(data_root_dir,"train_masks")))
        self.mask_list=[]
        self.image_list=[]
        self.box_list=[]
        self.cls_list=[]
        # directory containing all binary annotations
        for mask_path in self.mask_root_list:
            true_cls=int(mask_path.split('_')[-1].replace('.png', ''))
            if true_cls==0 or (train_cls!=cls and train_cls!=99):
                continue
            mask_path =osp.join(data_root_dir,"train_masks",mask_path)
            self.mask_list.append(mask_path)
            self.image_list.append('_'.join(mask_path.split('_')[:-1]).replace("train_masks","png_images")+'.png')
            self.box_list.append(mask_path.replace("train_masks","bbox").replace(".png",'.npy'))
            self.cls_list.append(true_cls)
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index], cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        #resize to 1024*1024*3,预处理已做过
        # image = ResizeLongestSide(1024).apply_image(image)
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.imread(self.mask_list[index], cv2.IMREAD_GRAYSCALE)
        # mask = mask[None,:,:]

        box  = np.load(self.box_list[index])
        true_cls = self.cls_list[index]
        return image, mask,box,true_cls,self.mask_list[index]

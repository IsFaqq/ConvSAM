import os
import os.path as osp
from glob import glob

import numpy as np
from tqdm import tqdm
import shutil
import random
from collections import defaultdict

class Seg_valdata:
    '''用于从训练集里面分离出来验证集'''
    def __init__(self, data_root_dir="/data/BTCV_Cervix/train_masks",
                 save_path ="/data/BTCV_Cervix/Valing/train_masks"):

        self.mask = sorted(os.listdir(data_root_dir))
        self.mask_list = [osp.join(data_root_dir,m) for m in self.mask]
        self.cls_mask_list =defaultdict(list)
        for mask_path in self.mask_list:
            cls = int(mask_path.split('_')[-1].replace('.png', ''))
            self.cls_mask_list[cls].append(mask_path)
        self.save_path =save_path
        os.makedirs(save_path,exist_ok=True)

    def seg(self,rate=0.1):
        '''每一类选rate比例的mask移动到验证集'''
        for cls,m_path in self.cls_mask_list.items():
            num = int(len(m_path)*rate)
            random_list = [random.randint(0, len(m_path)-1) for _ in range(num)]
            for index in tqdm(random_list):
                save_mask_name =osp.join(self.save_path,m_path[index].split('/')[-1])
                try:
                    shutil.move(m_path[index], save_mask_name)
                    print("文件已成功移动")
                except OSError as e:
                    print("文件已经移动！: %s" % e)
s =Seg_valdata("/mnt/afs/qianfaqiang/others/Abdomen/Training/train_masks","/mnt/afs/qianfaqiang/others/Abdomen/Valing/train_masks")
# s = Seg_valdata()
s.seg()





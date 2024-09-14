from segment_anything import sam_model_registry
import cv2
import torch
import numpy as np
from typing import Dict,List
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt
from model import ConvSAM_withUnetbox,MedSAM,ConvSAM_withUnet,ConvSAM_withResnetbox
from torch.nn import functional as F
import os,json
from monai.losses import DiceLoss
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
model_infect={
'ConvSAM_withUnetbox':ConvSAM_withUnetbox,
'ConvSAM_withUnet':ConvSAM_withUnet,
'ConvSAM_withResnetbox':ConvSAM_withResnetbox,
'MedSAM':MedSAM,
'SAM':MedSAM
}

type_infect ={
    1:'spleen',
    2:'kidney_right',
    3:'kidney_left',
    4:'gallbladder',
    5:'esophagus',
    6:'liver',
    7:'stomach',
    8:'aorta',
    9:'inferior_vena_cava',
    10:'portal_vein_and_splenic_vein',
    11:'pancreas',
    12:'adrenal_gland_right',
    13:'adrenal_gland_left'
    }
class Model_Infer():
    '''用于推理'''
    def __init__(self,
                 model_type :str="vit_b",
                 device ="cuda:0",
                 model_name ="SAM",
                 ckp_path:str ="/data/checkpoint/sam_vit_h_4b8939.pth",
                 ):
        self.device =device
        self.model_type =model_type
        self.model_name = model_name
        self.ckp_path = ckp_path
        self.sam = sam_model_registry[model_type](checkpoint="/mnt/afs/qianfaqiang/others/checkpoint/sam_vit_b_01ec64.pth").to(device)
        self.init_model()
    def init_model(self):
        self.ckp = torch.load(self.ckp_path)['model']
        self.model= model_infect[self.model_name](self.sam.image_encoder,self.sam.mask_decoder,self.sam.prompt_encoder).to(self.device)
        self.model.load_state_dict(self.ckp)
        self.model.eval()

    @torch.no_grad()
    def predict(self,image,box=None):
        if box is not None:
            box=np.array(box)
        # box= box.reshape(1,-1)
        H, W, C = image.shape
        # box_1024 = box / np.array([192, 156, 192, 156]) * 1024
        # image_tensor = np.transpose(image, (2, 0, 1))
        image_tensor = torch.as_tensor(image, dtype=torch.float).to(self.device)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        image_tensor = F.interpolate(
            image_tensor,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )

        low_res_logits = self.model(image_tensor,box)
        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        pelsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return pelsam_seg

class Eval_Infer():
    '''用于评估指标和保存图片'''
    def __init__(self,
                 model_type :str="vit_b",
                 device ="cuda:1",
                 model_name ="SAM",
                 ckp_path:str ="/data/checkpoint/sam_vit_h_4b8939.pth",
                 eval_path:str = "/data/BTCV_Cervix/Valing",
                 save_path:str="/data/BTCV_Cervix/Valing/out_images"):
        self.model_name = model_name
        self.predictor = Model_Infer(model_type,device,model_name,ckp_path)
        self.eval_path = eval_path
        self.save_path = save_path
        self.init_path()
        self.dice = DiceLoss(softmax=False)

    def init_path(self):
        self.mask_path_list = sorted(os.listdir(osp.join(self.eval_path,"train_masks")))
        self.mask_list=[]
        self.image_list=[]
        self.box_list=[]
        self.cls_list=[]
        self.cls_answer_dict=defaultdict(list)
        # directory containing all binary annotations
        for mask_path in self.mask_path_list:
            cls=int(mask_path.split('_')[-1].replace('.png', ''))
            if cls==0:
                continue
            mask_path =osp.join(self.eval_path,"train_masks",mask_path)
            self.mask_list.append(mask_path)
            self.image_list.append('_'.join(mask_path.split('_')[:-1]).replace("train_masks","png_images")+'.png')
            self.box_list.append(mask_path.replace("train_masks","bbox").replace(".png",'.npy'))
            self.cls_list.append(cls)

    def mask_save(self,mask_name,mask):
        os.makedirs(osp.join(self.save_path,f"masks_{self.model_name}"), exist_ok=True)
        mask_dir = osp.join(self.save_path,f"masks_{self.model_name}", mask_name)
        mask = Image.fromarray(mask)
        mask.save(mask_dir)

    def record(self):
        record_path = f"{self.save_path}/record_{self.model_name}.json"
        if os.path.exists(record_path):
            record_data = json.load(open(record_path))
        else:
            record_data={}
        record_data[self.model_name] = self.cls_answer_dict
        with open(record_path,'w',encoding='utf-8') as f:
            json.dump(record_data,f,ensure_ascii=False,indent=4)
        # df = pd.DataFrame([(key, value) for key, values in self.cls_answer_dict.items() for value in values], columns=['type', 'prediction'])

        # # Step 2: 绘制小提琴图
        # plt.figure(figsize=(10,8))
        # sns.violinplot(x='type', y='prediction', data=df)
        # plt.title('Violin Plot for Multiple Types')
        # plt.xlabel('Type')
        # plt.ylabel('Predictions')
        # plt.show()

    def infer(self,box_flag=True,eval_value=0.5,save_flag=True):
        '''推理指标并保存'''
        low_count = 0
        for (mask_path,image_path,box_path,cls) in tqdm(zip(self.mask_list,self.image_list,self.box_list,self.cls_list),total=len(self.mask_list)):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = (gt_mask / 255).astype(np.uint8)
            gt_mask = torch.as_tensor(gt_mask)
            gt_mask = gt_mask[None, :, :]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if box_flag==True:
                box = np.load(box_path)
            else:
                box=None
            box=np.expand_dims(box,0)
            mask = self.predictor.predict(image,box)
            mask = torch.as_tensor(mask)
            mask = mask[None, :, :]
            dice_value = 1.0 - self.dice(mask, gt_mask)
            mask=mask.squeeze(0).squeeze(0).detach().cpu().numpy()
            self.cls_answer_dict[cls].append(dice_value.item())
            if dice_value < eval_value:
                low_count += 1
            if save_flag==True:
                mask_name = mask_path.split('/')[-1]
                image_name = image_path.split('/')[-1]
                os.makedirs(osp.join(self.save_path, "png_images"), exist_ok=True)
                save_image_path =osp.join(self.save_path,"png_images",image_name)
                cv2.imwrite(save_image_path,image)
                self.mask_save(mask_name,mask*255)
        self.record()
        # mean_dice = sum(dice_list) / len(dice_list)
        # min_dice,max_dice =min(dice_list),max(dice_list)
        # print(f"平均dice{mean_dice},dice值低于{eval_value}的数量有{low_count}个")
        # print(f"最小dice{min_dice}，最大dice{max_dice}")
        # print(f"({mean_dice:.4f}+-{max(abs(mean_dice-min_dice),abs(mean_dice-max_dice)):.4f})")
        # return mean_dice
#/data/Abdomen/Training/model/SAM/20240911/SAM_b_20240911_bestdice_bs2_cls99.pth
#/data/Abdomen/Training/model/ConvSAM_withUnetbox/20240911/ConvSAM_withUnetbox_b_20240911_bestdice_bs2_cls99.pth
predict = Eval_Infer(model_type='vit_b',
                           model_name='ConvSAM_withResnetbox',
                           ckp_path=f"/mnt/afs/qianfaqiang/others/Abdomen/Training/model/ConvSAM_withResnetbox/20240912/ConvSAM_withResnetbox_b_20240912_bestdice_bs2_cls99.pth",
                           eval_path="/mnt/afs/qianfaqiang/others/Abdomen/Valing",
                           save_path="/mnt/afs/qianfaqiang/others/Abdomen/Valing/out_images"
                     )


predict.infer(save_flag=True)


# class Pelvic_Sam_Infer():
#     '''用于推理保存图片'''
#     def __init__(self,
#                  model_type :str="vit_b",
#                  device ="cuda:0",
#                  model_name ="sam",
#                  ckp_path:str ="/data/checkpoint/sam_vit_h_4b8939.pth",
#                  eval_path:str = "/data/BTCV_Cervix/Valing",
#                  save_path:str="/data/BTCV_Cervix/Valing/out_images",
#                  cls=1):
#         self.device =device
#         self.model_type =model_type
#         self.model_name = model_name
#         self.ckp_path = ckp_path
#         self.sam = sam_model_registry[model_type]().to(device)
#         self.eval_path = eval_path
#         self.init_model()
#         self.save_path = save_path
#         self.init_path(cls)
#         self.dice = DiceLoss()
#     def init_model(self):
#         self.ckp = torch.load(self.ckp_path)['model']
#         self.model= model_infect[self.model_name](self.sam.image_encoder,self.sam.mask_decoder,self.sam.prompt_encoder).to(self.device)
#         self.model.load_state_dict(self.ckp)
#         self.model.eval()
#
#     def init_path(self,val_cls):
#         self.cls=val_cls
#         self.mask_path_list = sorted(os.listdir(osp.join(self.eval_path,"train_masks")))
#         self.mask_list=[]
#         self.image_list=[]
#         self.box_list=[]
#         self.cls_list=[]
#         # directory containing all binary annotations
#         for mask_path in self.mask_path_list:
#             cls=int(mask_path.split('_')[-1].replace('.png', ''))
#             if cls==0 or (val_cls!=cls and val_cls!=99):
#                 continue
#             mask_path =osp.join(self.eval_path,"train_masks",mask_path)
#             self.mask_list.append(mask_path)
#             self.image_list.append('_'.join(mask_path.split('_')[:-1]).replace("train_masks","png_images")+'.png')
#             self.box_list.append(mask_path.replace("train_masks","bbox").replace(".png",'.npy'))
#             self.cls_list.append(cls)
#
#     def mask_save(self,image_name,mask):
#         os.makedirs(osp.join(self.save_path,f"masks_{self.model_name}_{self.cls}"), exist_ok=True)
#         mask_dir = osp.join(self.save_path,f"masks_{self.model_name}_{self.cls}", image_name)
#         mask = Image.fromarray(mask)
#         mask.save(mask_dir)
#
#     @torch.no_grad()
#     def predict(self,image,box=None):
#         if box:
#             box=np.array(box)
#         # box= box.reshape(1,-1)
#         H, W, C = image.shape
#         # box_1024 = box / np.array([192, 156, 192, 156]) * 1024
#         # image_tensor = np.transpose(image, (2, 0, 1))
#         image_tensor = torch.as_tensor(image, dtype=torch.float).to(self.device)
#         image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
#
#         image_tensor = F.interpolate(
#             image_tensor,
#             size=(1024, 1024),
#             mode="bilinear",
#             align_corners=False,
#         )
#
#         low_res_logits = self.model(image_tensor,box)
#         # low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
#         low_res_pred = F.interpolate(
#             low_res_logits,
#             size=(H, W),
#             mode="bilinear",
#             align_corners=False,
#         )  # (1, 1, gt.shape)
#         low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
#         pelsam_seg = (low_res_pred > 0).astype(np.uint8)
#         return pelsam_seg
#     def infer(self,box_flag=True,eval_value=0.1,save_flag=True):
#         '''推理指标并保存'''
#         dice_list = []
#         low_count = 0
#         for (mask_path,image_path,box_path) in tqdm(zip(self.mask_list,self.image_list,self.box_list)):
#             gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             gt_mask = (gt_mask / 255).astype(np.uint8)
#             gt_mask = torch.as_tensor(gt_mask)
#             gt_mask = gt_mask[None, :, :]
#             image = cv2.imread(image_path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             if box_flag==True:
#                 box = np.load(box_path)
#             else:
#                 box=None
#             mask = self.predict.predict(image,box)
#             mask = torch.as_tensor(mask)
#             mask = mask[None, :, :]
#             dice_value = 1.0 - self.dice(mask, gt_mask)
#             dice_list.append(dice_value.item())
#             if dice_value < eval_value:
#                 low_count += 1
#             if save_flag==True:
#                 image_name = image_path.split('/')[-1]
#                 os.makedirs(osp.join(self.save_path, "png_images"), exist_ok=True)
#                 save_image_path =osp.join(self.save_path,"png_images",image_name)
#                 cv2.imwrite(save_image_path,image)
#                 self.mask_save(image_name,mask)
#         mean_dice = sum(dice_list) / len(dice_list)
#         print(f"平均dice{mean_dice},dice值低于{eval_value}的数量有{low_count}个")
#         return mean_dice




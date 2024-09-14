'''
读入nii.gz的医学图像和标签，提取png和mask以及bbox，全是sam的输入格式1024*1024*3，box也做了转换
'''
import sys
sys.path.append("../..")
import os.path as osp
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
import torch
from torch.nn import functional as F
from tqdm import tqdm
import os
from  glob import glob
import nibabel as nib
from skimage import transform

# define the SAM model
# vit_mode = "h"
# if vit_mode == "h":
#     sam_checkpoint = "/apdcephfs_cq10/share_1567347/jingleqian/shenda/checkpoint/sam_vit_h_4b8939.pth"
# sam = sam_model_registry[f"vit_{vit_mode}"](checkpoint=sam_checkpoint)
# predictor = SamPredictor(sam)

# define data
class data_precess():
    def __init__(self,
                 data_root_dir = f"/data/data/BTCV_Cervix",
                 save_root_dir = f"/data/BTCV_Cervix_new",
                 sub_image_name = f"training_images",
                 sub_label_name = f"training_labels"
                 ):
        #read and write paths
        self.data_root_dir = data_root_dir
        self.train_path = os.path.join(data_root_dir,sub_image_name)
        self.train_label_path = os.path.join(data_root_dir,sub_label_name)
        self.save_root_dir = save_root_dir

        #nii文件
        self.nib_frame_list = sorted(glob(f'{self.train_path}/*.nii*'))
        self.nil_mask_list = sorted(glob(f'{self.train_label_path}/*.nii*'))
        #sam参数
        self.sam_size = 1024

        #统计
        self.len_nii_frame =0
        self.len_mask_frame =0
        self.index_label_num = [0] * 20


    def get_gt_mask_list(self,mask):
        '''
        每一个种类单独生成一个mask返回
        :param mask:
        :return:
        '''
        # 初始化一个空的字典，用于存储每个标签的 mask
        label_masks = {}

        # 遍历每个标签值（排除0）
        for label_value in np.unique(mask):
            if label_value == 0:
                continue

            # 创建一个与原始 mask 形状相同的零数组
            label_mask = np.zeros_like(mask)

            # 将当前标签值的像素设置为 1，其他像素保持为 0
            label_mask[mask == label_value] = 1

            # 将当前标签值的 mask 存储到字典中
            label_masks[label_value] = label_mask
        if label_masks=={}:
            label_masks[0] = np.zeros_like(mask)
        return label_masks
    def get_image_and_mask(self,frame_list,mask_list,nib_name):
        '''
        :param frame_list: 一张nii文件的图片，形状[w,h,slices]
        :param mask_list: 对应的nii文件的masks
        :return:
        '''
        self.len_nii_frame+=frame_list.shape[2]
        #暂留：是否做小物体筛选
        # self.filter_small_mask()

        # 找到非零切片,也就是有mask的图像部分
        # _, _,z_index = np.where(mask_list > 0)
        # z_index = np.unique(z_index)
        # self.len_mask_frame+=len(z_index)
        gt_masks = mask_list
        images = frame_list
        self.len_mask_frame += gt_masks.shape[2]
        # if (len(z_index)>0):
        #     gt_masks = mask_list[:,:,z_index]
        #     images = frame_list[:,:,z_index]
        #     print(f"切片数量：{frame_list.shape[2]}  标签可用:{len(z_index)}")
        # else:
        #     print("无可用切片")

        # #待定：窗宽增强
        # self.window_deal(images)

        #这里存储的是筛选和增强后的图像以及mask, spacing=images.GetSpacing()用于保存空间信息，可用spacing = img_nib.header.get_zooms()
        #比较占用空间，可选保存
        # os.makedirs(osp.join(self.save_root_dir,'origin_npz_files'),exist_ok=True)
        # np.savez_compressed(osp.join(self.save_root_dir,'origin_npz_files' ,nib_name+ 'choose.npz'), imgs=images,
        #                     gts=gt_masks)

        #转sam的输入形状
        for i,index in enumerate(range(gt_masks.shape[2])):
            image=images[:,:,i]
            mask = gt_masks[:,:,i]
            #处理图像，原始图像大小很乱
            # image_raw = self.imagegrayfloat_toRGBint(image)
            #这里也可以用reshape & padding的方式，待确定哪个好
            #image_sam = self.image_reshape_and_padding(image_raw)
            image_sam = transform.resize(
                image,
                (self.sam_size, self.sam_size),
                order=3,
                preserve_range=True,
                mode="constant",
                anti_aliasing=True,
            )
            image_sam=self.imagegrayfloat_toRGBint(image_sam)
            # save images
            os.makedirs(osp.join(self.save_root_dir, "png_images"),exist_ok=True)
            image_save_path = osp.join(self.save_root_dir, "png_images", f"{nib_name}_{index}.png")
            save_image = Image.fromarray(image_sam)
            save_image.save(image_save_path)

            #处理mask
            mask_class_dict = self.get_gt_mask_list(mask)
            #这里也可用用reshape & padding的方式然后set_mask方法，待定检验
            for label,mask in mask_class_dict.items():
                self.index_label_num[int(label)] +=1
                # print(f'标签大小{np.sum(mask)}')
                mask = transform.resize(
                    mask,
                    (self.sam_size, self.sam_size),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
                mask = np.uint8(mask)*255
                # print(f'放大后标签大小{np.sum(mask)}')

                #save mask
                mask_save_path = osp.join(self.save_root_dir, "train_masks",  f"{nib_name}_{index}_{int(label)}.png")
                os.makedirs(osp.join(self.save_root_dir, "train_masks"),exist_ok=True)
                mask_save = Image.fromarray(mask)
                mask_save.save(mask_save_path)

                if int(label) == 0:
                    continue
                #处理box
                object_coords = np.where(mask == 255)
                top_left_x = min(object_coords[1])
                top_left_y = min(object_coords[0])
                bottom_right_x = max(object_coords[1])
                bottom_right_y = max(object_coords[0])
                bbox =np.array([top_left_x,top_left_y,bottom_right_x,bottom_right_y])

                #save box
                box_save_path = osp.join(self.save_root_dir, "bbox", f"{nib_name}_{index}_{int(label)}.npy")
                os.makedirs( osp.join(self.save_root_dir, "bbox"), exist_ok=True)
                np.save(box_save_path,bbox)

            print(f"子任务完成{i+1}张")
        print(f"总任务完成{self.len_mask_frame}张")

    def check_integrity(self,frame_list,mask_list,nib_name):
        '''图片和mask对应的检查完整性'''
        if frame_list.shape != mask_list.shape:
            print(f"{nib_name}数据不完整：{nib_name}")
        else:
            print(f"{nib_name}数据完整")

    def process(self):
        '''
        主要处理函数，先获得图像和mask，然后存储和统计
        :return:
        '''
        for nib_frame_name, nib_mask_name in tqdm(zip(self.nib_frame_list, self.nil_mask_list), total=len(self.nib_frame_list)):
            frame_list = nib.load(nib_frame_name).get_fdata()
            mask_list = nib.load(nib_mask_name).get_fdata()
            nib_name = nib_frame_name.split('.nii')[0].split('/')[-1]
            self.check_integrity(frame_list, mask_list, nib_frame_name)
            self.get_image_and_mask(frame_list,mask_list,nib_name)
        print(f"共计{self.len_nii_frame} 实际可用：{self.len_mask_frame}")
        print(f'每个种类标签数量{self.index_label_num}')

    def set_mask(self,mask):
        '''
        sam的mask方法，先填充到256*256，然后复制三份使用sam的set_mask方法
        :param mask:
        :return:
        '''
        pass

    def imagegrayfloat_toRGBint(self,image):
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    def image_reshape_and_padding(self,image_raw):
        '''
        reshape and padding to size 256*256*3,然后使用sam的工具resize到1024*1024*3
        :param image_raw:
        :return:
        '''
        if image_raw.shape[0] <256 or image_raw.shape[1]<256:
            new_size = (256, 256,3)
            padded_image = np.zeros(new_size)
            padded_image[ :image_raw.shape[0], :image_raw.shape[1],:] = image_raw
            padded_image = np.uint8(image_raw)
        else:
            padded_image = image_raw
        deal_frame = ResizeLongestSide(1024).apply_image(padded_image)
        return deal_frame

    def mask_reshape_and_padding(self,mask_raw):
        one_mask = np.uint8(mask_raw)*255
        if one_mask.shape[0] < 256 or one_mask.shape[1] < 256:
            mask_size = (256,256)
            padded_mask = np.zeros(mask_size)
            padded_mask[ :one_mask.shape[0], :one_mask.shape[1]] = one_mask
            padded_mask = np.uint8(padded_mask)
        else:
            padded_mask = one_mask

    def filter_small_mask(self):
        '''
        用于过滤像素很小的mask
        :return:
        '''
        pass

    def window_deal(self,images):
        '''
        用于窗宽进行医学图像增强，主要是CT,是否用看效果
        :param images:
        :return:
        '''
        modality="CT"
        WINDOW_LEVEL = 40
        WINDOW_WIDTH =400
        if modality == "CT":
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(images, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            #选取50%-99.5%的值进行映射
            lower_bound, upper_bound = np.percentile(
                images[images > 0], 0.5
            ), np.percentile(images[images > 0], 99.5)
            image_data_pre = np.clip(images, lower_bound, upper_bound)
            #映射到0-255之间
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[images == 0] = 0

        return images

class testing_process():
    def __init__(self,
                 data_root_dir = f"/data/data/BTCV_Cervix",
                 save_root_dir = f"/data/BTCV_Cervix_new",
                 sub_image_name = f"testing_images"
                 ):
        self.data_root_dir = data_root_dir
        self.save_root_dir = save_root_dir
        os.makedirs(save_root_dir,exist_ok=True)

        self.test_path = os.path.join(data_root_dir, sub_image_name)

        self.nib_frame_list = sorted(glob(f'{self.test_path}/*.nii*'))
        self.len_nii_frame=0

        self.sam_size=1024

    def imagegrayfloat_toRGBint(self,image):
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def get_image(self,frame_list,nib_name):
        self.len_nii_frame+=frame_list.shape[2]

        # #待定：窗宽增强
        # self.window_deal(images)

        #转sam的输入形状
        for i in range(frame_list.shape[2]):
            image=frame_list[:,:,i]
            #处理图像，原始图像大小很乱
            # image_raw = self.imagegrayfloat_toRGBint(image)
            #这里也可以用reshape & padding的方式，待确定哪个好
            #image_sam = self.image_reshape_and_padding(image_raw)
            # image_sam = transform.resize(
            #     image,
            #     (self.sam_size, self.sam_size),
            #     order=3,
            #     preserve_range=True,
            #     mode="constant",
            #     anti_aliasing=True,
            # )
            # image_sam=self.imagegrayfloat_toRGBint(image)
            # save images
            os.makedirs(osp.join(self.save_root_dir, "png_images"),exist_ok=True)
            image_save_path = osp.join(self.save_root_dir, "png_images", f"{nib_name}_{i}.png")
            cv2.imwrite(image_save_path,image)
            # save_image = Image.fromarray(image_sam)
            # save_image.save(image_save_path)
            print(f"子任务完成{i + 1}张")

    def process(self):
        for nib_frame_name in tqdm(self.nib_frame_list, total=len(self.nib_frame_list)):
            frame_list = nib.load(nib_frame_name).get_fdata()
            nib_name = nib_frame_name.split('/')[-1].split('.nii')[0]
            self.get_image(frame_list, nib_name)
        print(f"共计{self.len_nii_frame}张测试图片")


def process(data_root_dir,save_root_dir,sub_image_name,process_mask=True,sub_label_name=None):
    '''process_mask决定生成训练集测试集还是验证集'''
    if process_mask:
        Data_proces = data_precess(data_root_dir, save_root_dir, sub_image_name, sub_label_name)
    else:
        Data_proces = testing_process(data_root_dir,save_root_dir,sub_image_name)
    Data_proces.process()


if __name__ =="__main__":
    # "/data/data/FLARE22Train/","/data/FLARE22Train/"
    #"/data/data/RawData/Training", "/data/Cervix_Data/Training", "img", "label"
    # process("/data/data/RawData/Training", "/data/Cervix_Data/Training","img", True,"label")
    process("/mnt/afs/qianfaqiang/others/abdomen","/mnt/afs/qianfaqiang/others/Abdomen/Training","imagesTr",True,"labelsTr")



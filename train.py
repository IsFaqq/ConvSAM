import json
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Cervix_dataset_four
from segment_anything import sam_model_registry,SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import monai
from loss import DiceLoss
from collections import defaultdict
from model import ConvSAM_withUnetbox,ConvSAM_withUnet,ConvSAM_withResnetbox,MedSAM
import cv2
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize

model_infect={
'ConvSAM_withUnetbox':ConvSAM_withUnetbox,
'ConvSAM_withUnet':ConvSAM_withUnet,
'ConvSAM_withResnetbox':ConvSAM_withResnetbox,
'MedSAM':MedSAM,
'SAM':MedSAM
}

def gettime():
    from datetime import datetime
    today = datetime.now().date()
    formatted_date = today.strftime('%Y%m%d')
    return formatted_date

if __name__ =="__main__":
    torch.cuda.empty_cache()
    torch.manual_seed(1999)
    #parm 
    train_time = gettime()
    # data_train_dir = "/data/BTCV_Cervix/Training"
    data_train_dir = "/mnt/afs/qianfaqiang/others/Abdomen/Training"
    batch_size = 2
    #每次修改模型、训练设备即可
    model_name = 'MedSAM'
    if model_name=="SAM":
        ckp_path ="/mnt/afs/qianfaqiang/others/checkpoint/sam_vit_b_01ec64.pth"
    else:
        ckp_path = "/mnt/afs/qianfaqiang/others/checkpoint/medsam_vit_b.pth"
    ckp_save_path =data_train_dir+f"/model/{model_name}/{train_time}"
    lr = 2e-4
    work_num = 2
    wd = 0.001
    num_epochs =50
    device ="cuda:0"
    train_cls= 99
    tensorboard_log_path = ckp_save_path + f"/runs_{train_cls}_0"
    if os.path.exists(tensorboard_log_path):
        time = int(tensorboard_log_path.split("_")[-1])
        tensorboard_log_path = "_".join(tensorboard_log_path.split("_")[:-1])
        tensorboard_log_path =f"{tensorboard_log_path}_{time}"

    os.makedirs(ckp_save_path,exist_ok=True)
    #初始化Convsam模型
    sam_model = sam_model_registry['vit_b'](checkpoint=ckp_path)
    conv_sam = model_infect[model_name](sam_model.image_encoder,sam_model.mask_decoder,sam_model.prompt_encoder).to(device)
    conv_sam.train()
    #tensorboard
    writer = SummaryWriter(log_dir=tensorboard_log_path)

    #读入数据
    # dataset = Cervix_dataset_four(data_root_dir=data_train_dir)
    dataset = Cervix_dataset_four(data_root_dir=data_train_dir,train_cls=train_cls)
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    np.save(os.path.join(tensorboard_log_path,f'testing_mask_list.npy'),np.array(test_dataset.dataset.mask_list))
    print(f'训练数据集数量:{len(train_dataset)} 测试数据集数量{len(test_dataset)}')


    #设置优化器和损失
    optimizer = torch.optim.AdamW(conv_sam.parameters(), lr=lr, weight_decay=wd)
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = DiceLoss(squared_pred=True,)
    loss_fn = monai.losses.DiceLoss(to_onehot_y=False,sigmoid=True, reduction="mean")
    eval_loss = monai.losses.DiceLoss(to_onehot_y=False, softmax=False)
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    criterion = nn.CrossEntropyLoss()

    T_max = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-7)
    #打印训练参数大小
    print("Number of total parameters: ",sum(p.numel() for p in conv_sam.parameters()),)  # 93735472
    print("Number of trainable parameters: ",sum(p.numel() for p in conv_sam.parameters() if p.requires_grad),)  # 93729252

    print(f"----------------start training----------------")
    epoch_losses = []
    # loss_log = []
    dice_epoch =[]
    best_dice=0
    for epoch in range(num_epochs):
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, )
        step_losses = []
        #train
        for step, (input_image,mask,box,cls,path) in enumerate(tqdm(train_data_loader,desc="training")):
            '''
            图片、掩码、box均是1024的大小
            input_image:[B,C,W,H]
            sam_feat:(B, 256, 64, 64)
            mask:[B,W,H]
            box:[B,4]
            cls:[B,1]
            '''
            box_torch = torch.as_tensor(box, dtype=torch.float32).to(device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            #deal image
            image = torch.as_tensor(input_image,dtype=torch.float).to(device)#（B,3，1024，1024）
            # writer.add_graph(pelvic_sam, input_image)
            pesam_pred = conv_sam(image,box)   #(B,H,W)

            ori_size= input_image.shape[-2:]
            if ori_size!=(1024,1024):
                pesam_pred = conv_sam.postprocess_masks(pesam_pred,ori_size,ori_size)
            # print(f'pesam_pred:{pesam_pred.shape}')

            gt_mask_resized = mask
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32).to(device)
            if len(gt_binary_mask.shape) != len(pesam_pred):
                gt_binary_mask = gt_binary_mask[:, None, :,:]

            # gt_binary_mask = gt_binary_mask.unsqueeze(0)
            #
            # pesam_pred = torch.gt(pesam_pred, 0).float()
            loss = loss_fn(pesam_pred, gt_binary_mask)+ ce_loss(pesam_pred, gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step_losses.append(loss.item())
            if step %600==0:
                print(f'Training epoch{epoch}  steps: {step}  loss: {loss.item()}')
            # loss_log.append({step+epoch*len(train_data_loader):mean(step_losses)})
            writer.add_scalar('loss',loss.item(),step+epoch*len(train_data_loader))
        epoch_losses.append(mean(step_losses))
        print(f'Training EPOCH: {epoch} Mean loss: {epoch_losses[-1]} lr:{scheduler.get_last_lr()}')
        writer.add_scalar('epoch_loss', epoch_losses[-1], epoch)
        # 每训练一个epoch，更新一次学习率
        scheduler.step()

        #tesing to choose model
        step_dice =[]
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, )
        for step,(input_image,mask,box,cls,path) in enumerate(tqdm(test_data_loader,desc="testning")):
            box_torch = torch.as_tensor(box, dtype=torch.float32).to(device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            #deal image
            image = torch.as_tensor(input_image,dtype=torch.float).to(device)#（B,3，1024，1024）

            with torch.no_grad():
                pesam_pred = conv_sam(image,box)
                ori_size = input_image.shape[-2:]
                if ori_size!=(1024,1024):
                    pesam_pred= conv_sam.postprocess_masks(pesam_pred, ori_size, ori_size)
            gt_mask_resized = mask
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32).to(device)
            # gt_binary_mask = gt_binary_mask.unsqueeze(0)
            if len(gt_binary_mask.shape) != len(pesam_pred.shape):
                gt_binary_mask = gt_binary_mask[:, None, :,:]

            pesam_pred = (pesam_pred>0.5).type(torch.float32)
            dice_value = 1.0 - eval_loss(pesam_pred, gt_binary_mask)
            step_dice.append(dice_value.item())
            # if step % 20 == 0:
            #     print(f'Testing EPOCH{epoch}  steps: {step}  dice: {dice_value.item()}')
        dice_epoch.append(mean(step_dice))
        print(f'Testing EPOCH: {epoch} Mean dice: {dice_epoch[-1]}')
        writer.add_scalar('dice', dice_epoch[-1], epoch)

        ## save the latest model
        checkpoint = {
            "model": conv_sam.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        os.makedirs(ckp_save_path,exist_ok=True)
        torch.save(checkpoint, os.path.join(ckp_save_path, f"{model_name}_b_{train_time}_latest_bs{batch_size}_cls{train_cls}.pth"))
        ## save the best model
        if dice_epoch[-1] > best_dice:
            best_dice =dice_epoch[-1]
            checkpoint = {
                "model": conv_sam.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(ckp_save_path, f"{model_name}_b_{train_time}_bestdice_bs{batch_size}_cls{train_cls}.pth"))
    print(os.path.join(ckp_save_path, f"{model_name}_b_{train_time}_bestdice_bs{batch_size}_cls{train_cls}.pth"))
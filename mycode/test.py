
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from model import MCNN, weights_init
from dataset import CrowdDataset
from tqdm import tqdm

net = MCNN()
device_ids = [0]
net = net.cuda(device_ids[0])
net = nn.DataParallel(net, device_ids = device_ids)
net.load_state_dict(torch.load("data/saved_models/net_900_5.01753.pth"))

img_path = 'data/test_data/sesame/IMG_20200423_192253.jpg'

with torch.no_grad():
    net.eval()
    
    #
    ori_img = plt.imread(img_path)

    #
    ds_rows=int(ori_img.shape[0]//4)
    ds_cols=int(ori_img.shape[1]//4)
    img = cv2.resize(ori_img,(ds_cols * 4, ds_rows * 4))
    img = img.transpose((2,0,1)) # convert to order (channel,rows,cols)
    img=img[np.newaxis,:,:,:]
    img_tensor=torch.tensor(img,dtype=torch.float)
    img_tensor_cuda = img_tensor.float().cuda(device_ids[0])
    
    #
    density_map_cuda = net(img_tensor_cuda)
    density_map = density_map_cuda.data.cpu().numpy()
    count = np.sum(density_map)
    print(count)
    plt.imsave('data/_density_map.bmp',density_map[0,0,:,:])
    
    #
    gt_dmap=np.load(img_path.replace('.jpg','.npy'))
    gt_count = np.sum(gt_dmap)   
    print(gt_count)
    plt.imsave('data/_gt_dmap.bmp',gt_dmap)

    #
    # img = cv2.resize(ori_img,(density_map.shape[1], density_map.shape[0]))
    # dst = cv2.addWeighted(img,0.5,density_map[0,0,:,:],0.5,0)
    # plt.imsave('data/_dst.bmp',gt_dmap)
    cv2.putText(ori_img,'ground truth: ' + str(int(gt_count)),(50,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.putText(ori_img,'count: ' + str(int(count)),(50,250),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    plt.imsave('data/_img.bmp',ori_img)






    
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


class CrowdDataset(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self,img_dir,gt_downsample=1):
        '''
        img_dir: the root path of img.
        gt_downsample: default is 1, denote that the output of deep-model is the same size as input image.
        '''
        self.img_dir=img_dir
        self.gt_downsample=gt_downsample

        # self.img_names=[filename for filename in os.listdir(img_root) \
        #                    if os.path.isfile(os.path.join(img_root,filename))]

        scene_paths = [os.path.join(img_dir,scene) for scene in os.listdir(img_dir)]
        self.img_paths = []
        for scene_path in scene_paths:
            for file in os.listdir(scene_path):
                if file.split('.')[-1] == 'bmp' and file.find('_.bmp') != -1:
                    self.img_paths.append(os.path.join(scene_path, file))

        self.sample_num=len(self.img_paths)

    def __len__(self):
        return self.sample_num

    def __getitem__(self,index):
        assert index <= (self.sample_num - 1), 'index range error'
        img_path=self.img_paths[index]
        img=plt.imread(img_path)
        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)

        gt_dmap=np.load(img_path.replace('_.bmp','npy'))
        # print(gt_dmap.sum())
        # if self.gt_downsample > 1: # to downsample image and density-map to match deep-model.
        #     ds_rows=int(img.shape[0]//self.gt_downsample)
        #     ds_cols=int(img.shape[1]//self.gt_downsample)
        #     img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
        #     img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)
        #     gt_dmap=cv2.resize(gt_dmap,(ds_cols,ds_rows))
        #     gt_dmap=gt_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample

        ds_rows=int(img.shape[0]//self.gt_downsample)
        ds_cols=int(img.shape[1]//self.gt_downsample)
        img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
        img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)
        gt_dmap=cv2.resize(gt_dmap,(ds_cols,ds_rows))
        gt_dmap=gt_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample
        
        img_tensor=torch.tensor(img,dtype=torch.float)
        gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
        # print(gt_dmap_tensor.sum())

        # return img_tensor,gt_dmap_tensor
        return img,gt_dmap


# test code
if __name__=="__main__":
    img_dir='data/train_data'
    dataset=CrowdDataset(img_dir, gt_downsample=4)
    for i,(img,gt_dmap) in enumerate(dataset):
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(gt_dmap)
        # plt.figure()
        print(img.shape,gt_dmap.shape, gt_dmap.sum())

        if i == 3:
            break;
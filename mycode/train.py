import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from mycode.model import MCNN, weights_init
from mycode.dataset import CrowdDataset
from tqdm import tqdm


# hyper_param
start_step = 1
end_step = 2000
lr = 0.0001
momentum = 0.9
val_interval = 50
save_interval = 100
batch_size = 32

# dataset
train_dir = 'data/train_data'
test_dir = 'data/test_data'

dataset_train = CrowdDataset(img_dir=train_dir, gt_downsample=4)
dataloader_train = torch.utils.data.DataLoader(dataset = dataset_train,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2,
                                                drop_last=True)

dataset_test = CrowdDataset(img_dir=test_dir, gt_downsample=4)
dataloader_test = torch.utils.data.DataLoader(dataset = dataset_test,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=2,
                                                drop_last=True)

# model
net = MCNN()
weights_init(net, dev=0.01)
device_ids = [i for i in range(0,8)]
net = nn.DataParallel(net, device_ids = device_ids)
net = net.cuda(device_ids[0])

criterion = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr = lr, momentum = momentum, weight_decay = 1e-4)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# training
output_dir = 'data/saved_models'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_loss = 0
step_cnt = 0
re_cnt = False

for epoch in range(start_step, end_step+1):  
    print('Epoch {}/{}'.format(epoch, end_step))

    net.train()
    for i, data in enumerate(dataloader_train):                
        img,gt_dmap = data
        img_cuda = img.float().cuda(device_ids[0])
        gt_dmap_cuda = gt_dmap.float().cuda(device_ids[0])
        density_map_cuda = net(img_cuda)
        loss = criterion(density_map_cuda, gt_dmap_cuda)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % val_interval == 0: 
            gt_dmap = gt_dmap.numpy()           
            gt_count = np.sum(gt_dmap)    
            density_map = density_map_cuda.data.cpu().numpy()
            count = np.sum(density_map)
            print('gt_count: ', gt_count, 'count: ', count, 'loss: ', count - gt_count)    
            plt.imsave('data/density_map.bmp',density_map[0,0,:,:])
            plt.imsave('data/gt_dmap.bmp',gt_dmap[0,0,:,:])
            img = img.numpy().transpose((0,2,3,1))
            plt.imsave('data/img.bmp',img[0,:,:,:])
          
    if epoch % save_interval == 0:
        with torch.no_grad():
            net.eval()
            loss_sum = 0.0
            for i, data in enumerate(dataloader_test):
                img,gt_dmap = data
                img_cuda = img.float().cuda(device_ids[0])
                gt_dmap_cuda = gt_dmap.float().cuda(device_ids[0])
                density_map_cuda = net(img_cuda)
                    
                gt_dmap = gt_dmap.numpy()           
                gt_count = np.sum(gt_dmap)    
                density_map = density_map_cuda.data.cpu().numpy()
                count = np.sum(density_map)
                loss_sum += abs(count - gt_count)
                
            print("loss_avg: %.3f" %(loss_sum / (i + 1)))
        
            torch.save(net.state_dict(), "%s/net_%03d_%.3f.pth" %(output_dir, epoch, loss_sum / (i + 1)))

    lr_scheduler.step()

        
    


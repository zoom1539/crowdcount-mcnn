'''
Modify test_data/65/001.json first, or will raise error.
'''
import os
import json
import  xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import cv2

    
if __name__ == '__main__':
    phase_list = ['test','train']
    for phase in phase_list:
        scene_dir = 'data/' + phase + '_data'
        for scene in os.listdir(scene_dir):
            img_dir = os.path.join(scene_dir,scene)
            '''process each image'''
            for name in tqdm(os.listdir(img_dir)):
                if name.split('.')[-1] != 'bmp' or name.find('_.bmp') != -1:
                    continue
                img_path = os.path.join(img_dir,name)
                img = plt.imread(img_path)

                ds_rows=int(img.shape[0]//2)
                ds_cols=int(img.shape[1]//2)
                img = cv2.resize(img,(ds_cols,ds_rows))
                img = img[30:715,270:960,:]
                save_path = img_path.replace('bmp','_.bmp')
                plt.imsave(save_path, img)

                

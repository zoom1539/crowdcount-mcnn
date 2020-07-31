import os
import json
import  xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import cv2

def generate_densitymap(image, circles):
    '''
    Use fixed size kernel to construct the ground truth density map 
    for Fudan-ShanghaiTech. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [x,y,width,height]. 
    sigma: the sigma of gaussian_kernel to simulate a head. 
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]

    # quantity of heads in the image
    circles_num = len(circles)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
    for circle in circles:
        x = min(int(round(circle[0])),image_w-1)
        y = min(int(round(circle[1])),image_h-1)
        r = int(round(circle[2]))
        r = max(r, 1)
        r = min(r, 10)
        # width = int(round(point[2]))
        # height = int(round(point[3]))
        point2density = np.zeros((image_h, image_w), dtype=np.float32)
        point2density[y,x] = 1
        # densitymap += gaussian_filter(point2density, sigma=(width,height), mode='constant')
        # densitymap += gaussian_filter(point2density, sigma=15, mode='constant')
        densitymap += gaussian_filter(point2density, sigma=r, mode='constant')

    # densitymap = densitymap / densitymap.sum() * circles_num
    return densitymap    

    
if __name__ == '__main__':
    phase_list = ['test','train']
    for phase in phase_list:
        scene_dir = 'data/' + phase + '_data'
        for scene in os.listdir(scene_dir):
            img_dir = os.path.join(scene_dir,scene)
            '''process each image'''
            for name in tqdm(os.listdir(img_dir)):
                if name.split('.')[-1] != 'bmp' or name.find('_.bmp') == -1:
                    continue
                img_path = os.path.join(img_dir,name)
                json_path = img_path.replace('_.bmp','json')
                npy_path = img_path.replace('_.bmp','npy')
                img = plt.imread(img_path)

                #
                circles = []
                with open(json_path, 'r', encoding = 'gbk') as f:
                    annotation = json.loads(f.read())
                    for shape in annotation['shapes']:
                        circle = [i * 0.5 for i in shape['points'][0]]
                        circle[0] -= 270
                        circle[1] -= 30
                        center = np.array(shape['points'][0] ) / 2
                        edge = np.array(shape['points'][1] ) / 2
                        r = np.linalg.norm(edge - center) 
                        circle.append(r)

                        circles.append(circle)

                        #
                        # cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (255,0,0))
                        # save_circle_path = img_path.replace('jpg','__.jpg')
                        # plt.imsave(save_circle_path, img)

                '''generate density map'''
                densitymap = generate_densitymap(img, circles)
                np.save(npy_path,densitymap)

                save_path = img_path.replace('_.bmp','_dm.bmp')
                plt.imsave(save_path, densitymap)

import os
import json
import  xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import cv2

# img size
# row:30:715
# col:270:960
if __name__ == '__main__':
    phase = 'train'
    scene_dir = 'data/' + phase + '_data'
    with open("data/yolo_data/train.txt", "w") as f:
        for scene in os.listdir(scene_dir):
            img_dir = os.path.join(scene_dir,scene)
            '''process each image'''
            for name in tqdm(os.listdir(img_dir)):
                if name.find('_.bmp') == -1:
                    continue
                img_path = os.path.join(img_dir,name)
                f.write(img_path + "\n")

                json_path = img_path.replace('_.bmp','json')
                circles = []
                with open(json_path, 'r', encoding = 'gbk') as f_json:
                    annotation = json.loads(f_json.read())
                    for shape in annotation['shapes']:
                        if shape['shape_type'] != "circle":
                            continue

                        circle = [i * 0.5 for i in shape['points'][0]]
                        circle[0] -= 270
                        circle[1] -= 30
                        center = np.array(shape['points'][0] ) / 2
                        edge = np.array(shape['points'][1] ) / 2
                        r = np.linalg.norm(edge - center) 
                        circle.append(r)
                        circles.append(circle)
                    
                label_path = img_path.replace('bmp','txt')
                label_path = label_path.replace('data/train_data/diff_size_colony','data/yolo_data')
                print(label_path)
                with open(label_path, "w") as f_lable:
                    for circle in circles:
                        f_lable.write("0 ")
                        f_lable.write(str(circle[0] / 690))
                        f_lable.write(" ")
                        f_lable.write(str(circle[1] / 685))
                        f_lable.write(" ")
                        f_lable.write(str(circle[2] * 2 / 690))
                        f_lable.write(" ")
                        f_lable.write(str(circle[2] * 2 / 685))
                        f_lable.write("\n")
                # input()

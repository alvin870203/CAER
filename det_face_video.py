import copy
import os
import shutil

import cv2
import face_alignment
import numpy as np
import torch
from PIL import Image

from util import make_dir


def main():

    paths = [
        './data/CAER/CAER/CAER/train.txt',
        './data/CAER/CAER/CAER/validation.txt',
        './data/CAER/CAER/CAER/test.txt',
    ]
    result_paths = [
        './data/CAER/CAER/CAER-Face/train.txt',
        './data/CAER/CAER/CAER-Face/validation.txt',
        './data/CAER/CAER/CAER-Face/test.txt',
    ]

    # cuda for CUDA
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')

    
    for i, path in enumerate(paths):
        path_list, label_list = load_txt(path)

        type_file = path.split('/')[-1]
        result_path = make_dir(result_paths[i].replace(type_file, ''), False)
        result_path = make_dir(result_path + type_file.replace('.txt', ''), True)

        result_file = result_paths[i]
        
        for i, file_path in enumerate(path_list):
            
            label = label_list[i]
            label_dir = make_dir(os.path.join(result_path, label), False)
            save_dir  = make_dir(os.path.join(label_dir, file_path.split('/')[-1].replace('.avi','')), False)

            read_video(result_file, fa, file_path, save_dir)

def read_video(result_file, fa, path, save_dir):

    cap = cv2.VideoCapture(path)
    if (cap.isOpened() == False):
        print('error =>', path)
    
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            image_path = os.path.join(save_dir, str(count) + '.png')
            cv2.imwrite(image_path, frame)
            det_result = det_face(fa, frame, image_path)

            if det_result is not None:
                # print(result_file, image_path, det_result)
                with open(result_file, 'a') as f:
                    f.write(image_path + ' ' + det_result[0] + ' ' + str(det_result[1][0]) + ',' +  str(det_result[1][1]) + ',' +  str(det_result[1][2]) + ',' + str(det_result[1][3]) + '\n')
                
            count += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

def det_face(fa, image, file_path):

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    or_image  = Image.fromarray(rgb_image)

    face_path = file_path.replace('.png', '_face.png')

    try:
        preds = fa.get_landmarks(rgb_image)

        max_area = 0
        max_pred = []

        if preds is not None:
            for i, pred in enumerate(preds):

                xmin = int(min(pred[:, 0]))
                xmax = int(max(pred[:, 0]))
                ymin = int(min(pred[:, 1]))
                ymax = int(max(pred[:, 1]))

                w = xmax - xmin
                h = ymax - ymin

                b_size = max(w, h)
                mid_x = (xmax + xmin) /2
                mid_y = (ymax + ymin) /2

                area = w*h
                if area > max_area:
                    max_area = area
                    max_pred = [mid_x - b_size*0.5, mid_y - b_size*0.5, mid_x + b_size*0.5, mid_y + b_size*0.5]

                    or_image.crop(max_pred).save(face_path)

            return (face_path, max_pred)
        else:
            print(face_path + " => not found face")
            return None
    except:
        print('error => ' + face_path)
        return None

def load_txt(path):

    path_list  = []
    label_list = []
    with open(path,'r') as f:

        data_lines = f.readlines()
        for line in data_lines:

            line = line.replace('\n', '')
            line = line.replace('\\', '/')
            line = line.split(' ')

            path_list.append(line[0])
            label_list.append(line[1])

    return path_list, label_list

if __name__ == "__main__":
    main()

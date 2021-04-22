import os
import os.path as osp
import shutil
import face_alignment
from PIL import Image
import numpy as np
import copy
import time

def main():
    # cuda for CUDA
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')

    org_path = './data/AVEC14/'
    org_type = ['Training.txt', 'Testing.txt']
    org_name = 'Image'

    tgt_path = './data/AVEC14/FaceImage/'
    tgt_name = 'FaceImage'

    log_file = './log.txt'
    
    if osp.isdir(tgt_path):
        shutil.rmtree(tgt_path)
    os.mkdir(tgt_path)

    for data_type in org_type:
        data_path = osp.join(org_path, data_type)
        data_item = load_txt(data_path)

        for org_path_dir in data_item:
            org_path_dir = org_path_dir.split(' ')[1]
            tgt_path_dir = make_tree_dir(tgt_path, org_path_dir.replace(org_name, tgt_name))

            st = time.time()
            for image_name in os.listdir(org_path_dir):
                image = Image.open(osp.join(org_path_dir, image_name))
                np_image = np.array(image)

                # det face
                preds = fa.get_landmarks(np_image)
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

                    image.crop(max_pred).save(osp.join(tgt_path_dir, image_name))
                else:
                    f = open(log_file, "a")
                    f.write(osp.join(org_path_dir, image_name) + '\n')
                    f.close()
            ed = time.time()
            print(org_path_dir, '=>', tgt_path_dir, 'running time: {:.4f}'.format(ed - st), 'done!')

def make_tree_dir(header_path, path):
    items = path.split('/')[4:]

    result_path = header_path
    for item in items:
        result_path = osp.join(result_path, item)
        if osp.isdir(result_path) == False:
            os.mkdir(result_path)
    return result_path

def load_txt(path):
    result = []
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        result.append(line.replace('\n', ''))

    return result

if __name__ == "__main__":
    main()
    # for data_type in os.listdir(org_path):
    #     org_data_dir = osp.join(org_path, data_type)
    #     tgt_data_dir = osp.join(tgt_path, data_type)
    #     os.mkdir(tgt_data_dir)

    #     for label_dir in os.listdir(org_data_dir):
            
    #         org_label_dir = osp.join(org_data_dir, label_dir)
    #         tgt_label_dir = osp.join(tgt_data_dir, label_dir)
    #         os.mkdir(tgt_label_dir)

    #         for image_file in os.listdir(org_label_dir):
                
    #             image = Image.open(osp.join(org_label_dir, image_file))
    #             np_image = np.array(image)

    #             # det face
    #             preds = fa.get_landmarks(np_image)
    #             max_area = 0
    #             max_pred = []

    #             if preds is not None:
    #                 for i, pred in enumerate(preds):

    #                     xmin = int(min(pred[:, 0]))
    #                     xmax = int(max(pred[:, 0]))
    #                     ymin = int(min(pred[:, 1]))
    #                     ymax = int(max(pred[:, 1]))

    #                     w = xmax - xmin
    #                     h = ymax - ymin

    #                     b_size = max(w, h)
    #                     mid_x = (xmax + xmin) /2
    #                     mid_y = (ymax + ymin) /2

    #                     area = w*h
    #                     if area > max_area:
    #                         max_area = area
    #                         max_pred = [mid_x - b_size*0.5, mid_y - b_size*0.5, mid_x + b_size*0.5, mid_y + b_size*0.5]

    #                 image.crop(max_pred).save(osp.join(tgt_label_dir, image_file))
    #             else:
    #                 f = open(log_file, "a")
    #                 f.write(osp.join(org_label_dir, image_file) + '\n')
    #                 f.close()
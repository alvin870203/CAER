import os
import os.path as osp
import shutil
import face_alignment
from PIL import Image
import numpy as np
import copy

if __name__ == "__main__":
    
    # cuda for CUDA
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')

    org_path = './data/CAER/CAER-S/CAER-S'
    tgt_path = './data/CAER/CAER-S/CAER-S-FACE'
    log_file = './log.txt'
    
    if os.path.isdir(tgt_path):
        shutil.rmtree(tgt_path)
    os.mkdir(tgt_path)

    for data_type in os.listdir(org_path):
        org_data_dir = osp.join(org_path, data_type)
        tgt_data_dir = osp.join(tgt_path, data_type)
        os.mkdir(tgt_data_dir)

        for label_dir in os.listdir(org_data_dir):
            
            org_label_dir = osp.join(org_data_dir, label_dir)
            tgt_label_dir = osp.join(tgt_data_dir, label_dir)
            os.mkdir(tgt_label_dir)

            for image_file in os.listdir(org_label_dir):
                
                image = Image.open(osp.join(org_label_dir, image_file))
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

                    image.crop(max_pred).save(osp.join(tgt_label_dir, image_file))
                else:
                    f = open(log_file, "a")
                    f.write(osp.join(org_label_dir, image_file) + '\n')
                    f.close()
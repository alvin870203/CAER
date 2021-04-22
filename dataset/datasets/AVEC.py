import os
import os.path as osp
import random

import numpy as np

from .bases import BaseImageDataset


class AVEC(BaseImageDataset):

    # dataset_dir = 'AVEC14'
    def __init__(self, root='./data/', **kwargs):
        super(AVEC, self).__init__()

        self.mode = 'Freeform'    
        self.dataset_dir= root
        self.train_path = osp.join(self.dataset_dir, 'Training.txt')
        self.test_path  = osp.join(self.dataset_dir, 'Testing.txt')

        # label template
        # self.label_template = [
        #     [i for i in range(0, 14)],
        #     [i for i in range(14, 20)],
        #     [i for i in range(20, 29)],
        #     [i for i in range(29, 45)]
        # ]
        self.label_template = [i for i in range(45)]
        self.num_classes    = len(self.label_template)

        self.train = self._load_file(self.train_path)
        self.test  = self._load_file(self.test_path)
        
        self.train_num_d_levels, self.train_num_data = self.get_avec_imagedata_info(self.train)
        self.test_num_d_levels,  self.test_num_data  = self.get_avec_imagedata_info(self.test)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # level|   # data |")
        print("  ----------------------------------------")
        print("  train    |  {:5d} | {:8d} |".format(self.num_classes, self.train_num_data))
        print("  test     |  {:5d} | {:8d} |".format(self.num_classes, self.test_num_data))
        print("  ----------------------------------------")

    def _load_file(self, file):

        dataset = []
        f = open(file, 'r') 
        lines = f.readlines()
        ratio = 5
        d_values = []
        for i, line in enumerate(lines):
            line = line.replace('\n', '').split(' ')
            video_path = line[0]
            image_path = line[1]
            d_val      = int(line[2])
            # d_level    = -1
            # for level, val_list in enumerate(self.label_template):
            #     if d_val in val_list:
            #         d_level = level

            # if d_level == -1:
            #     print(video_path, image_path, d_level)
            #     assert i == -1
            # else:
            if self.mode in image_path:
                    # print(video_path, image_path, d_val)
                dataset.append((image_path, image_path.replace('/Image', '/FaceImage'), d_val))
                d_values.append(d_val)
                    # for context_file in os.listdir(image_path):
                    #     context_file = os.path.join(image_path, context_file)
                    #     face_file    = context_file.replace('/Image', '/FaceImage')
                    #     if os.path.isfile(face_file):
                    #         # print(context_file, face_file)
                    #         video_item.append((video_path, context_file, face_file, d_val, d_level))
                    #     else:
                    #         video_item.append((video_path, context_file, context_file, d_val, d_level))
                    # video_item = np.array(video_item)

                    # select_idx = np.random.choice(np.arange(video_item.shape[0]), video_item.shape[0] // ratio)
                    # select_idx = np.sort(select_idx)

                    # for item in video_item[select_idx, :]:
                    #     dataset.append((item[1], item[2], int(item[3]), int(item[4])))
                # print(video_path, image_path, len(os.listdir(image_path)), d_val, d_level)
        
        # print(d_values, max(d_values))
        # print(len(dataset))
        return dataset
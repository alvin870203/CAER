import os
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self, cfg, dataset, f_transform = None, c_transform=None):
        self.c_transform = c_transform
        self.f_transform = f_transform
        self.pre_load_option    = cfg.DATALOADER.PRE_LOAD
        self.c_size = (cfg.INPUT.CONTEXT_SIZE[1], cfg.INPUT.CONTEXT_SIZE[0])
        self.f_size = (cfg.INPUT.FACE_SIZE[1], cfg.INPUT.FACE_SIZE[0])

        if self.pre_load_option:
            self.dataset = self.pre_load(dataset)
        else:
            self.dataset = dataset

    def pre_load(self, dataset):
        result = []
        for context_path, face_path, id in dataset:
            context = read_image(context_path).resize(self.c_size)
            face    = read_image(face_path).resize(self.f_size)
            result.append((context, face, id))
        return result

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        if self.pre_load_option == False:
            context_path, face_path, id = self.dataset[index]
            context = read_image(context_path)
            face    = read_image(face_path)
        else:
            context, face, id = self.dataset[index]

        if (self.c_transform is not None) and (self.f_transform is not None):
            context = self.c_transform(context)
            face    = self.f_transform(face)

        return context, face, id

class DepressionVideoDataset(Dataset):
    def __init__(self, cfg, dataset, f_transform = None, c_transform=None, mode='train'):
        self.c_transform = c_transform
        self.f_transform = f_transform

        self.mode = mode
        if mode == 'train':
            self.sampled_frames = cfg.DATALOADER.NUM_INSTANCE 
        else:
            self.sampled_frames = cfg.DATALOADER.NUM_INSTANCE * 10
        self.fps = 10
        self.dataset = self.build_frame_idx(dataset)

    def build_frame_idx(self, dataset):

        data_dict = []
        for item in dataset:
            # load frame
            context_dir = item[0]
            face_dir    = item[1]
            label       = item[2]
            # check frames
            frame_len   = len(os.listdir(context_dir))
            context_lists = []
            face_lists    = []
            for i in range(frame_len):

                context_path = osp.join(context_dir, str(i) + '.png')
                face_path    = osp.join(face_dir, str(i) + '.png')

                context_lists.append(context_path)
                if osp.isfile(face_path):
                    face_lists.append(face_path)
                else:
                    face_lists.append(context_path)
                
            context_lists = np.array(context_lists)
            face_lists    = np.array(face_lists)
            index         = np.arange(0, frame_len)
            data_dict.append({'context': context_lists, 'face': face_lists, 'frame_index': index, 'label': label})
        return data_dict

    def __len__(self):
        return len(self.dataset)

    def select_index(self, index):

        # freg_idx = index.shape[0]//self.freg
        # print(freg_idx)


        # assert i == -1
        sampled_idx = np.random.choice(index, self.sampled_frames, replace=False)
        sampled_idx = np.sort(sampled_idx)
        return sampled_idx

    def load_frame_from_path(self, path_list, transform):
        tensor = []
        for path in path_list:
            tensor.append(transform(read_image(path)))
        return torch.stack(tensor)

    def __getitem__(self, index):
        item = self.dataset[index]

        val  = item['label']
        sampled_idx = self.select_index(item['frame_index'])
        context_path = item['context'][sampled_idx]
        face_path    = item['face'][sampled_idx]
        if (self.c_transform is not None) and (self.f_transform is not None):
            context = self.load_frame_from_path(context_path, self.c_transform)
            face    = self.load_frame_from_path(face_path,    self.f_transform)
        return context, face, val

class VideoDataset(Dataset):
    def __init__(self, cfg, dataset, f_transform = None, c_transform=None):
        ## dataset => load frame
        self.dataset = self.build_video_frame_index(dataset)
        self.c_transform = c_transform
        self.f_transform = f_transform
        self.k = cfg.DATALOADER.FRAME_NUM

    def build_video_frame_index(self, dataset):
        # item => video_path, image_path, d_val, d_level
        data_list = []
        for item in dataset:
            # load frame
            frame_path = item[1]
            # context
            frame_list = os.listdir(item[1])
            frame_num  = len(frame_list)

            context_frame_list = []
            face_frame_list = []
            # face
            for frame_id in range(0, frame_num):
                frame_name = str(frame_id) + '.png'

                context_path = osp.join(frame_path, frame_name)

                if osp.isfile(context_path):
                    context_frame_list.append(context_path)
                    face_path    = context_path.replace('Image','FaceImage')
                    if osp.isfile(face_path):
                        face_frame_list.append(face_path)
                    else:
                        face_frame_list.append(context_path)


            # change list to array
            context_frame_list = np.array(context_frame_list)
            face_frame_list    = np.array(face_frame_list)

            frame_idx  = np.arange(frame_num)
            data_list.append({'video': item[0], 
                              'video_path': item[1], 
                              'context_frame': context_frame_list, 
                              'face_frame': face_frame_list, 
                              'indices': frame_idx, 
                              'val': item[2], 
                              'level': item[3]})
        return data_list

    def __len__(self):
        return len(self.dataset)
    
    def load_frame_from_path(self, path_list, transform):
        tensor = []
        for path in path_list:
            tensor.append(transform(read_image(path)))
        return torch.stack(tensor)
        
    def __getitem__(self, index):
        item = self.dataset[index]
        # uniform sample
        sampled_idx = np.random.choice(item['indices'], self.k, replace=False)
        sampled_idx = np.sort(sampled_idx)
        context_video_frame = item['context_frame'][sampled_idx]
        face_video_frame = item['face_frame'][sampled_idx]
        
        # temp: need changing to image tensor
        # return context_video_frame, face_video_frame, item['val'], item['level']

        if (self.c_transform is not None) and (self.f_transform is not None):
            context = self.load_frame_from_path(context_video_frame, self.c_transform)
            face    = self.load_frame_from_path(face_video_frame, self.f_transform)

        return context, face, item['val'], item['level']



        # context = read_image(context_path)
        # face    = read_image(face_path)

        # if (self.c_transform is not None) and (self.f_transform is not None):
        #     context = self.c_transform(context)
        #     face    = self.f_transform(face)

        # return context, face, id

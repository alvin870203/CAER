import os
import os.path as osp
from .bases import BaseVideoDataset

class CAER(BaseVideoDataset):

    dataset_dir = 'CAER/CAER'
    def __init__(self, root='./data/', **kwargs):
        super(CAER, self).__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_path  = osp.join(self.dataset_dir, 'train.txt')
        self.valid_path  = osp.join(self.dataset_dir, 'validation.txt')
        self.test_path   = osp.join(self.dataset_dir, 'test.txt')

        self.train = self.load_txt(self.train_path)
        self.valid = self.load_txt(self.valid_path)
        self.test  = self.load_txt(self.test_path)

        self.num_train_eids, self.num_train = self.get_videodata_info(self.train)
        self.num_valid_eids, self.num_valid = self.get_videodata_info(self.valid)
        self.num_test_eids,  self.num_test  = self.get_videodata_info(self.test)

        self.print_dataset_statistics(self.train, self.valid, self.test)

    def load_txt(self, path):

        data_list = []

        with open(path, 'r') as f:
            data_txt = f.readlines()

            for line in data_txt:
                
                line  = line.replace('\n', '')
                line  = line.split(' ')
                path  = line[0]
                label = line[-1]
                data_list.append((path, label))

        return data_list
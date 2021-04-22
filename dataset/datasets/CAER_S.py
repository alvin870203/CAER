import os
import os.path as osp
from .bases import BaseImageDataset

class CAER_S(BaseImageDataset):

    dataset_dir = 'CAER-S/CAER-S'
    def __init__(self, root='./data/', **kwargs):
        super(CAER_S, self).__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir  = osp.join(self.dataset_dir, 'test')

        self.label_template = self._make_id_template(self.train_dir)

        self.train = self._process_dir(self.train_dir)
        self.test  = self._process_dir(self.test_dir)

        self.num_train_eids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_eids, self.num_test_imgs   = self.get_imagedata_info(self.test)

        self.print_dataset_statistics(self.train, self.test)

    def _make_id_template(self, train_dir):
        
        label_template = []
        for label in os.listdir(train_dir):
            label_template.append(label)
        return label_template

    def _process_dir(self, dir_path):
        
        dataset = []
        no_face = 0
        for key in os.listdir(dir_path):
            eid = self.label_template.index(key)
            image_dir = osp.join(dir_path, key)
            for image_path in os.listdir(image_dir):
                
                context_path = osp.join(image_dir, image_path)
                face_path    = context_path.replace('CAER-S/CAER-S', 'CAER-S/CAER-S-FACE')
                if os.path.isfile(face_path):
                    dataset.append((context_path, face_path, eid))
                else:
                    no_face += 1
                    dataset.append((context_path, context_path, eid))
        return dataset
import os
import os.path

import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings


class MyDataset(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split='train', data_fraction=None,
                 build_probabilities=False):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().my_dataset_dir if root is None else root
        super().__init__('MyDataset', root, image_loader)
        self.split = split

        # Keep a list of all classes
        self.data_path, self.anno_path, self.visible_path = self._build_paths(split)
        self.sequence_list = self._build_sequence_list()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        if build_probabilities:
            self.class_probabilities_list = self._build_class_probabilities()
            self.seq_probs = self._build_seq_probabilities()
        self.build_probabilities = build_probabilities

    def _build_paths(self, split):
        data_path = os.path.join(self.root, split)
        anno_path = os.path.join(self.root, "bounding_boxes")
        visible_path = os.path.join(self.root, "out_of_view")
        return data_path, anno_path, visible_path

    def _read_bb_anno(self, anno_path):

        gt = pandas.read_csv(anno_path, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, outside_path):
        with open(outside_path, 'r', newline='') as f:
            outsides = torch.ByteTensor([not int(v) for v in list(csv.reader(f))[0]])
        return outsides

    def get_sequence_info(self, seq_id):
        video_name = self.sequence_list[seq_id] + ".txt"
        anno_path = os.path.join(self.anno_path, video_name)
        outside_path = os.path.join(self.visible_path, video_name)
        bboxes: torch.tensor = self._read_bb_anno(anno_path)
        visible: torch.tensor = self._read_target_visible(outside_path)
        valid = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)
        return {'bbox': bboxes, 'valid': valid,
                'visible': visible}

    def get_name(self):
        return "custom_dataset"

    def get_split(self):
        return self.split

    def _build_sequence_list(self):

        sequence_list = os.listdir(self.data_path)
        return sequence_list

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.data_path, seq_name)

    def _get_frame(self, seq_path, frame_id):
        img_path = os.path.join(seq_path, '{:08}.jpg'.format(frame_id + 1))
        img = self.image_loader(img_path)

        return img

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        if frame_ids is None:
            frame_ids = range(len(os.listdir(seq_path)))
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta


if __name__ == '__main__':
    a = MyDataset()
    data_path, anno_path, visible_path = a._build_paths()
    print(data_path)

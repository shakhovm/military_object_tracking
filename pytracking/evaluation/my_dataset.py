import csv

import numpy as np
import pandas
import torch
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os
from PIL import Image
from pathlib import Path
from ltr.admin.environment import env_settings


class MyDataset(BaseDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, root=None, split='train'):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.root = env_settings().my_dataset_dir if root is None else root
        self.split = split

        # Keep a list of all classes
        self.data_path, self.anno_path, self.visible_path = self._build_paths(split)

        self.sequence_list = self._build_sequence_list()
        # self.sequence_list_id =
        self.split = split

        # self.mask_path = None
        # if self.vos_mode:
        #     self.mask_path = self.env_settings.got10k_mask_path

    def _build_sequence_list(self):

        sequence_list = os.listdir(self.data_path)
        return sequence_list

    def _build_paths(self, split):
        data_path = os.path.join(self.root, split)
        anno_path = os.path.join(self.root, "bounding_boxes")
        visible_path = os.path.join(self.root, "out_of_view")
        return data_path, anno_path, visible_path

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _read_bb_anno(self, anno_path):

        gt = pandas.read_csv(anno_path, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return np.array(gt)

    def _read_target_visible(self, outside_path):
        with open(outside_path, 'r', newline='') as f:
            outsides = np.array([not int(v) for v in list(csv.reader(f))[0]])
        return outsides

    def _construct_sequence(self, sequence):

        video_name = sequence + ".txt"
        anno_path = os.path.join(self.anno_path, video_name)
        outside_path = os.path.join(self.visible_path, video_name)
        bboxes: torch.tensor = self._read_bb_anno(anno_path)
        visible: torch.tensor = self._read_target_visible(outside_path)
        valid = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)


        # anno_path =
        # anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        #
        # ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        path_dir = os.path.join(self.data_path, sequence)
        frame_list = [frame for frame in os.listdir(path_dir)]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(path_dir, frame) for frame in frame_list]

        # masks = None
        # if self.vos_mode:
        #     seq_mask_path = '{}/{}'.format(self.mask_path, sequence_name)
        #     masks = [self._load_mask(Path(self._get_anno_frame_path(seq_mask_path, f[:-3] + 'png'))) for f in
        #              frame_list[0:1]]

        return Sequence(sequence, frames_list, 'my_dataset', bboxes.reshape(-1, 4))#ground_truth_rect.reshape(-1, 4))

    @staticmethod
    def _load_mask(path):
        if not path.exists():
            print('Error: Could not read: ', path, flush=True)
            return None
        im = np.array(Image.open(path))
        im = np.atleast_3d(im)[..., 0]
        return im

    def _get_anno_frame_path(self, seq_path, frame_name):
        return os.path.join(seq_path, frame_name)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'ltrval':
            with open('{}/got10k_val_split.txt'.format(self.env_settings.dataspec_path)) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list

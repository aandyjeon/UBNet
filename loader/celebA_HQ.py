import sys
import os

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from torch import Tensor

import math

import numpy as np
import pandas as pd

import cv2
import json

import random

import torch
from torch.utils.data import Dataset, DataLoader


class CelebA_HQ(Dataset):
    def __init__(self, data: dict = None, root: str = '/data', txt_file: str = None, 
                 image_size: int = 224) -> None:

        self._image_size: int = image_size
        self._root: str       = root
        
        with open(txt_file) as txt_file:
            self._image_list: List[str] = txt_file.read().splitlines()

        random.shuffle(self._image_list)

    def __getitem__(self, index: int) -> torch.Tensor:
        
        _image_dir: str = self._image_list[index].split(',')[0]
        self.label: int = int(self._image_list[index].split(',')[1])

        self.image = cv2.resize(cv2.imread(os.path.join(self._root,_image_dir)), \
                                dsize=(self._image_size, self._image_size), interpolation=cv2.INTER_CUBIC) \
                                .transpose(2,0,1)/255.

        return torch.FloatTensor(self.image), self.label

    def __len__(self) -> int:
        return len(self._image_list)

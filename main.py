import sys, json
sys.path.append('~/Unbiased_Learning_on_Unknown_Bias')
# -*- coding: utf-8 -*-

import os
import random
import json

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

from torch.backends import cudnn

from option import get_option
from trainer import Trainer
from utils import save_option

from loader.celebA_HQ import CelebA_HQ


def backend_setting(option):
    log_dir = os.path.join(option.save_dir, option.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = option.gpu

    if option.random_seed is None:
        option.random_seed = random.randint(1,10000)
    torch.manual_seed(option.random_seed)

    if torch.cuda.is_available() and not option.cuda:
        print('[WARNING] GPU is available, but not use it')

    if option.cuda:
        torch.cuda.manual_seed_all(option.random_seed)
        cudnn.benchmark = option.cudnn_benchmark

def main():
    option = get_option()
    backend_setting(option)
    trainer = Trainer(option)

    if option.data == 'CelebA-HQ':
        
        train_dataset  = CelebA_HQ(root = '/data', txt_file = 'datasets/CelebA-HQ/train.txt')
        ub1_valid_dataset  = CelebA_HQ(root = '/data', txt_file = 'datasets/CelebA-HQ/ub1_val.txt')
        ub2_valid_dataset  = CelebA_HQ(root = '/data', txt_file = 'datasets/CelebA-HQ/ub2_val.txt')
        
        train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=option.batch_size,
                                num_workers=option.num_workers)

        ub1_valid_loader = data.DataLoader(dataset=ub1_valid_dataset,
                                       batch_size=option.batch_size,
                                       num_workers=option.num_workers)

        ub2_valid_loader = data.DataLoader(dataset=ub2_valid_dataset,
                                       batch_size=option.batch_size,
                                       num_workers=option.num_workers)

        print(f"train_dataset: {len(train_dataset)} | ub1_valid_dataset: {len(ub1_valid_dataset)} | ub2_valid_dataset: {len(ub2_valid_dataset)}")
        print(f"train_loader: {len(train_loader)} | ub1_valid_loader: {len(ub1_valid_loader)} | ub2_valid_loader: {len(ub2_valid_loader)}")

        if option.is_train:
            save_option(option)
            trainer.train(train_loader=train_loader, ub1_val_loader=eb1_valid_loader, ub2_val_loader = ub2_valid_loader)

if __name__ == '__main__': main()

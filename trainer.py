# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from torchvision import models
from torchsummary import summary

import model.orthonet as orthonet

import time
import os
import math
import sys

import numpy as np

from tqdm import tqdm
from utils import logger_setting
#from torch.utils.tensorboard import SummaryWriter
#
#writer = SummaryWriter('runs/celebA-HQ')


class Trainer(object):
    def __init__(self, option):
        self.option = option

        self._build_model()
        self._set_optimizer()
        self.logger = logger_setting(option.exp_name, option.save_dir)


    def _build_model(self):
        if self.option.data == 'CelebA-HQ': #for adding other dataset
            if self.option.model == 'vgg11':
                print("[MODEL] vgg11")
                if self.option.imagenet_pretrain:
                    print("ImageNet Pre-trained")
                    self.net = models.vgg11(pretrained = True)
                else:
                    self.net = models.vgg11(pretrained = False)
                self.net.classifier[6] = nn.Linear(4096, self.option.n_class)
            elif self.option.model == 'resnet18':
                print("[MODEL] resnet18")
                if self.option.imagenet_pretrain:
                    self.net = models.resnet18(pretrained = True)
                else:
                    self.net = models.resnet18(pretrained = False)
                self.net.fc = nn.Linear(512, self.option.n_class)
            elif self.option.model == 'alexnet':
                print("[MODEL] alexnet")
                if self.option.imagenet_pretrain:
                    self.net = models.alexnet(pretrained=True)
                else:
                    self.net = models.alexnet(pretrained = False)
                self.net.classifier[6] = nn.Linear(4096, self.option.n_class)


        if self.option.orthonet:
            if self.option.data == 'CelebA-HQ':
                self.orthnet = orthonet.OrthoNet(num_classes=self.option.n_class)
                self.loss_orth = nn.CrossEntropyLoss(ignore_index=255)
                self._load_model()

        else: 
            self.loss = nn.CrossEntropyLoss(ignore_index=255)

        if self.option.cuda:

            if self.option.orthonet:
                self.orthnet.cuda()
                self.net.cuda()
                self.loss_orth.cuda()
                print(f"[PARAMETER:ORTHONET]: {self._count_parameters(self.orthnet)}")
                print(f"[PARAMETER:BASELINE]: {self._count_parameters(self.net)}")
            else:
                self.net.cuda()
                self.loss.cuda()
                print(f"[PARAMETER:BASELINE]: {self._count_parameters(self.net)}")


    def _count_parameters(self,model):
        
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def _set_optimizer(self):
        if self.option.orthonet:
            self.optim_orth = optim.Adam(filter(lambda p: p.requires_grad, self.orthnet.parameters()), lr=self.option.lr,  weight_decay=self.option.weight_decay)
        else:
            self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr, weight_decay=self.option.weight_decay)

        lr_lambda = lambda step: self.option.lr_decay_rate ** (step // self.option.lr_decay_period)

        if self.option.orthonet:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optim_orth, lr_lambda=lr_lambda, last_epoch=-1)
        else:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=-1)


    @staticmethod
    def _weights_init_xavier(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_normal_(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data, gain=1.0)            


    def _initialization(self):
        if self.option.is_train:
            if self.option.imagenet_pretrain:
            #initializing only the classifier
                print('[INITIALIZED]')
                if self.option.model == 'vgg11' or self.option.model == 'alexnet':
                    self.net.classifier.apply(self._weights_init_xavier)
                elif self.option.model == 'resnet18':
                    self.net.fc.apply(self._weights_init_xavier)
            else:
            #initializing all the parameters
                self.net.apply(self._weights_init_xavier)

            if self.option.orthonet:
                self.orthnet.apply(self._weights_init_xavier)

            if self.option.use_pretrain:
                if self.option.checkpoint is not None:
                    self._load_model()
                else:
                    print("[WARNING] no pre-trained model")


    def _mode_setting(self, is_train=True):

        if is_train:
            if self.option.orthonet:
                self.orthnet.train()
                self.net.train()
            else: self.net.train()

        else:
            if self.option.orthonet:
                self.orthnet.eval()
                self.net.eval()
            else: self.net.eval()


    def _train_step(self, data_loader, step):
        self._mode_setting(is_train=True)
        
        loss_sum = 0.
        loss_orth_sum = 0.
        loss_conv_sum = 0.
        loss_trans_sum = 0.
        total_num_train = 0
        for i, (images,labels) in enumerate(tqdm(data_loader)):
            
            images = self._get_variable(images)
            labels = self._get_variable(labels)
            total_num_train += images.shape[0]

            if self.option.orthonet:

                if self.option.data == 'CelebA-HQ':
                    """
                    Training OrthoNet
                    """
                    if self.option.model == 'vgg11':
                        new_classifier = nn.Sequential(*list(self.net.children())[0])
                        extractor_1 = nn.Sequential(*list(new_classifier.children())[:3]).cuda()
                        extractor_2 = nn.Sequential(*list(new_classifier.children())[:6]).cuda()
                        extractor_3 = nn.Sequential(*list(new_classifier.children())[:11]).cuda()
                        extractor_4 = nn.Sequential(*list(new_classifier.children())[:16]).cuda()
                        extractor_5 = nn.Sequential(*list(new_classifier.children())[:21]).cuda()
                    elif self.option.model == 'resnet18':
                        extractor_1 = nn.Sequential(*list(self.net.children())[:4]).cuda()
                        extractor_2 = nn.Sequential(*list(self.net.children())[:5]).cuda()
                        extractor_3 = nn.Sequential(*list(self.net.children())[:6]).cuda()
                        extractor_4 = nn.Sequential(*list(self.net.children())[:7]).cuda()
                        extractor_5 = nn.Sequential(*list(self.net.children())[:8]).cuda()
                    elif self.option.model == 'alexnet':
                        new_classifier = nn.Sequential(*list(self.net.children())[0])
                        extractor_1 = nn.Sequential(*list(new_classifier.children())[:3]).cuda()
                        extractor_2 = nn.Sequential(*list(new_classifier.children())[:6]).cuda()
                        extractor_3 = nn.Sequential(*list(new_classifier.children())[:13]).cuda()
                        
                    if self.option.model == 'alexnet':
                        for param in extractor_1.parameters():
                            param.requires_grad = False
                        for param in extractor_2.parameters():
                            param.requires_grad = False
                        for param in extractor_3.parameters():
                            param.requires_grad = False

                        feature_1 = extractor_1.forward(images)
                        feature_2 = extractor_2.forward(images)
                        feature_3 = extractor_3.forward(images)

                        out = {}
                        out['out1'] = feature_1
                        out['out2'] = feature_2
                        out['out3'] = feature_3
                    else:
                        for param in extractor_1.parameters():
                            param.requires_grad = False
                        for param in extractor_2.parameters():
                            param.requires_grad = False
                        for param in extractor_3.parameters():
                            param.requires_grad = False
                        for param in extractor_4.parameters():
                            param.requires_grad = False
                        for param in extractor_5.parameters():
                            param.requires_grad = False
    
                        feature_1 = extractor_1.forward(images)
                        feature_2 = extractor_2.forward(images)
                        feature_3 = extractor_3.forward(images)
                        feature_4 = extractor_4.forward(images)
                        feature_5 = extractor_5.forward(images)
    
                        out = {}
                        out['out1'] = feature_1
                        out['out2'] = feature_2
                        out['out3'] = feature_3
                        out['out4'] = feature_4
                        out['out5'] = feature_5

                    self.optim_orth.zero_grad()
                    if self.option.ablation:
                        pred_label_orth = self.orthnet(out)
                        loss_orth = self.loss_orth(pred_label_orth, labels)
                        loss_orth_sum += loss_orth
                    else:
                        pred_label_orth, loss_conv, loss_trans = self.orthnet(out)
                        loss_orth = self.loss_orth(pred_label_orth, labels)
                        loss_orth_sum += loss_orth
                        loss_conv_sum += loss_conv
                        loss_trans_sum += loss_trans

                    loss_orth.backward()
                    self.optim_orth.step()
                    
                        
            else:
                """
                Training Baseline
                """
                #vgg training
                self.optim.zero_grad()
                if self.option.data == 'CelebA-HQ':
                    pred_label = self.net(images)
                loss = self.loss(pred_label, torch.squeeze(labels))
                loss_sum += loss
                loss.backward()
                self.optim.step()
                    

        if self.option.orthonet:    
            msg = f"[TRAIN] ORTH LOSS : {loss_orth_sum/len(data_loader)} LOSS_CONV : {loss_conv_sum/total_num_train} LOSS_TRANS : {loss_trans_sum/total_num_train}"
        else: 
            msg = f"[TRAIN] BASE LOSS : {loss_sum/len(data_loader)}"
        self.logger.info(msg)


    def _validate(self, data_loader, valid_type=None, step=None):
        self._mode_setting(is_train=False)

        if not self.option.is_train:
            print("[VALIDATING]")
            self._initialization()
            if self.option.checkpoint is not None:
                self._load_model()
            else:
                print("No trained model")
                sys.exit()

        total_num_correct = 0.
        total_num_correct_orth = 0.
        total_num_test = 0.
        total_loss = 0.
        total_loss_orth = 0.
        total_loss_conv = 0.
        total_loss_trans = 0.
        total_loss_trans = 0.

        for i, (images,labels) in enumerate(tqdm(data_loader)):
            
            images = self._get_variable(images)
            labels = self._get_variable(labels)
            
            batch_size = images.shape[0]
            total_num_test += batch_size

            if self.option.orthonet:
                self.optim_orth.zero_grad()
                
                if self.option.data == 'CelebA-HQ':
                    if self.option.model == 'vgg11':                    
                        new_classifier = nn.Sequential(*list(self.net.children())[0])
                        extractor_1 = nn.Sequential(*list(new_classifier.children())[:3]).cuda()
                        extractor_2 = nn.Sequential(*list(new_classifier.children())[:6]).cuda()
                        extractor_3 = nn.Sequential(*list(new_classifier.children())[:11]).cuda()
                        extractor_4 = nn.Sequential(*list(new_classifier.children())[:16]).cuda()
                        extractor_5 = nn.Sequential(*list(new_classifier.children())[:21]).cuda()

                    elif self.option.model == 'resnet18':
                        extractor_1 = nn.Sequential(*list(self.net.children())[:4]).cuda()
                        extractor_2 = nn.Sequential(*list(self.net.children())[:5]).cuda()
                        extractor_3 = nn.Sequential(*list(self.net.children())[:6]).cuda()
                        extractor_4 = nn.Sequential(*list(self.net.children())[:7]).cuda()
                        extractor_5 = nn.Sequential(*list(self.net.children())[:8]).cuda()

                    elif self.option.model == 'alexnet':
                        new_classifier = nn.Sequential(*list(self.net.children())[0])
                        extractor_1 = nn.Sequential(*list(new_classifier.children())[:3]).cuda()
                        extractor_2 = nn.Sequential(*list(new_classifier.children())[:6]).cuda()
                        extractor_3 = nn.Sequential(*list(new_classifier.children())[:13]).cuda()
                        
                    if self.option.model == 'alexnet':
                        for param in extractor_1.parameters():
                            param.requires_grad = False
                        for param in extractor_2.parameters():
                            param.requires_grad = False
                        for param in extractor_3.parameters():
                            param.requires_grad = False

                        feature_1 = extractor_1.forward(images)
                        feature_2 = extractor_2.forward(images)
                        feature_3 = extractor_3.forward(images)

                        out = {}
                        out['out1'] = feature_1
                        out['out2'] = feature_2
                        out['out3'] = feature_3

                    else:
     
                        for param in extractor_1.parameters():
                            param.requires_grad = False
                        for param in extractor_2.parameters():
                            param.requires_grad = False
                        for param in extractor_3.parameters():
                            param.requires_grad = False
                        for param in extractor_4.parameters():
                            param.requires_grad = False
                        for param in extractor_5.parameters():
                            param.requires_grad = False
    
                        feature_1 = extractor_1.forward(images)
                        feature_2 = extractor_2.forward(images)
                        feature_3 = extractor_3.forward(images)
                        feature_4 = extractor_4.forward(images)
                        feature_5 = extractor_5.forward(images)
                        
                        out = {}
                        out['out1'] = feature_1
                        out['out2'] = feature_2
                        out['out3'] = feature_3
                        out['out4'] = feature_4
                        out['out5'] = feature_5

                    pred_label_orth, loss_conv, loss_trans = self.orthnet(out)
                    loss_orth = self.loss_orth(pred_label_orth, labels)
                    total_num_correct_orth += self._num_correct(pred_label_orth,labels,topk=1).data
                    total_loss_orth += loss_orth.data*batch_size
                    total_loss_conv += loss_conv
                    total_loss_trans += loss_trans
                    
    
            if not self.option.orthonet:   
                self.optim.zero_grad()
                pred_label = self.net(images)
                loss = self.loss(pred_label, labels)
                
                total_num_correct += self._num_correct(pred_label,labels,topk=1).data
                total_loss += loss.data*batch_size

        if self.option.orthonet:

            avg_loss_orth = total_loss_orth/total_num_test
            avg_acc_orth = total_num_correct_orth/total_num_test

            if valid_type != None:
                msg = f"[EVALUATION - {valid_type}] LOSS : {avg_loss_orth}, ACCURACY : {avg_acc_orth}"
            else:
                msg = f"[EVALUATION] LOSS : {avg_loss_orth}, ACCURACY : {avg_acc_orth} LOSS_CONV : {total_loss_conv/total_num_test} LOSS_TRANS : {total_loss_trans/total_num_test}"
   
                   
        if not self.option.orthonet:   

            avg_loss = total_loss/total_num_test
            avg_acc = float(total_num_correct)/total_num_test
            if valid_type != None:
                msg = f"[EVALUATION - {valid_type}] LOSS : {avg_loss}, ACCURACY : {avg_acc}"
            else:
                msg = f"[EVALUATION] LOSS : {avg_loss}, ACCURACY : {avg_acc}"
        
        self.logger.info(msg)


    def _num_correct(self,outputs,labels,topk=1):
        _, preds = outputs.topk(k=topk, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).sum()
        return correct


    def _accuracy(self, outputs, labels):
        batch_size = labels.size(0)
        _, preds = outputs.topk(k=1, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).float().sum(0, keepdim=True)
        accuracy = correct.mul_(100.0 / batch_size)
        return accuracy


    def _save_model(self, step):
        if self.option.orthonet:
            torch.save({
                'step': step,
                'optim_state_dict': self.optim_orth.state_dict(),
                'net_state_dict': self.orthnet.state_dict()
            }, os.path.join(self.option.save_dir,self.option.exp_name, f'checkpoint_step_{step}.pth'))
        else:
            torch.save({
                'step': step,
                'optim_state_dict': self.optim.state_dict(),
                'net_state_dict': self.net.state_dict()
            }, os.path.join(self.option.save_dir,self.option.exp_name, f'checkpoint_step_{step}.pth'))

        print(f'[SAVE] checkpoint step: {step}')


    def _load_model(self):
        ckpt = torch.load(self.option.checkpoint)
        self.net.load_state_dict(ckpt['net_state_dict'],strict=False)


    def train(self, train_loader, eb1_val_loader=None, eb2_val_loader=None):
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()

        self._mode_setting(is_train=True)
        start_epoch = 0
        for step in range(start_epoch, self.option.max_step):
            self._train_step(train_loader,step)
            self.scheduler.step()

            if step == 1 or step % self.option.save_step == 0 or step == (self.option.max_step-1):

                if self.option.data == 'CelebA-HQ':
                    if eb1_val_loader is not None and eb2_val_loader is not None:
                        self._validate(eb1_val_loader, step = step)
                        self._validate(eb2_val_loader, valid_type = 'eb2', step = step)
                        
                self._save_model(step)
                

    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)

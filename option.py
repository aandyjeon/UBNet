# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--exp_name',   required=True,                 help='experiment name')

parser.add_argument('--data',             default='afhq',    type=str,   help='afhq MNIST')
parser.add_argument('--n_class',          default=2,        type=int,   help='number of classes')
parser.add_argument('--input_size',       default=224,       type=int,   help='input size')
parser.add_argument('--batch_size',       default=32,        type=int,   help='mini-batch size')
parser.add_argument('--lr',               default=0.001,     type=float, help='initial learning rate')
parser.add_argument('--lr_decay_rate',    default=0.1,       type=float, help='lr decay rate')
parser.add_argument('--lr_decay_period',  default=10,        type=int,   help='lr decay period')
parser.add_argument('--max_step',         default=10,        type=int,   help='maximum step for training')
parser.add_argument('--seed',             default=2,         type=int,   help='seed index')
parser.add_argument('--model',            default='vgg11',               help='vgg11|resnet18|alexnet')
parser.add_argument('--weight_decay',     default=0.0005,    type=float, help='Adam optimizer weight decay')


parser.add_argument('--ubnet',         action='store_true',        help='whether using orthonet')
parser.add_argument('--checkpoint',       default=None,               help='checkpoint to resume')
parser.add_argument('--log_step',         default=50,     type=int,   help='step for logging in iteration')
parser.add_argument('--save_step',        default=1,      type=int,   help='step for saving in epoch')
parser.add_argument('--data_dir',         default='./',               help='data directory')
parser.add_argument('--save_dir',         default='./',               help='save directory for checkpoint')
parser.add_argument('--data_split',       default='train',            help='data split to use')
parser.add_argument('--use_pretrain',     default=False,              help='whether it use pre-trained parameters if exists')
parser.add_argument('--imagenet_pretrain',action='store_true',        help='whether it train baseline or unlearning')
 

parser.add_argument('--random_seed',                      type=int,   help='random seed')
parser.add_argument('--num_workers',      default=4,      type=int,   help='number of workers in data loader')
parser.add_argument('--cudnn_benchmark',  default=True,   type=bool,  help='cuDNN benchmark')


parser.add_argument('--cuda',             action='store_true',        help='enables cuda')
parser.add_argument('--is_train',         action='store_true',        help='whether it is training')
parser.add_argument('--is_valid',         action='store_true',        help='whether it is validation')
parser.add_argument('--gpu',              default='0',                help='which number of gpu used')


def get_option():
    option = parser.parse_args()
    return option

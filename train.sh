#train basline
python main.py -e celeba --is_train --is_valid --cuda --imagenet_pretrain --data CelebA-HQ --n_class 2 --lr 0.0001 --max_step 2 --gpu 1 --model vgg11 --lr_decay_period 10 --lr_decay_rate 0.1

#train orthonet
python main.py -e celeba_orth --is_train --is_valid --orthonet --cuda --use_pretrain True --checkpoint celeba/checkpoint_step_1.pth --data CelebA-HQ --n_class 2 --lr 0.0001 --max_step 2 --lr_decay_rate 0.1 --lr_decay_period 10 --model vgg11 --gpu 1

#train basline
python base_model/main_base.py -e celebA_train --imagenet_pretrain --data_dir dataset --save_dir exp --data CelebA-HQ --is_train --model vgg11 --batch_size=32 --max_step=20 --lr=0.0001 --cuda --gpu=0 --lr_scheduler step --lr_decay_period=10

#train orthonet
python main.py -e celebA_ubnet_train --is_train --ubnet --cuda --checkpoint exp/celebA_train/checkpoint_step_19.pth --data CelebA-HQ --data_dir dataset --save_dir exp --lr=0.0001 --max_step=20 --gpu=0 --batch_size=32 --model vgg11 --lr_scheduler step --lr_decay_period=10

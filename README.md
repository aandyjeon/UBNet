# A Conservative Approach for Unbiased Learning on Unknown Biases (Accepted on CVPR2022)

## Abstract

Although convolutional neural networks (CNNs) achieve state-of-the-art in image classification, recent works address their unreliable predictions due to their excessive dependence on biased training data. Existing unbiased modeling postulates that the bias in the dataset is obvious to know, but it is actually unsuited for image datasets including countless sensory attributes. To mitigate this issue, we present a new scenario that does not necessitate a predefined bias. Under the observation that CNNs do have multi-variant and unbiased representations in the model, we propose a conservative framework that employs this internal information for unbiased learning. Specifically, this mechanism is implemented via hierarchical features captured along the multiple layers and orthogonal regularization. Extensive evaluations on public benchmarks demonstrate our method is effective for unbiased learning.

<p align="center">
<img src="https://user-images.githubusercontent.com/50168126/160750955-f3d0621a-787f-4c66-8746-ce197cf78ebf.png" width="600" height="480">
</p>
**Categorization of Data Bias Problems** with an example of gender prediction biased in hair length. The most limited case, unknown bias, is what we address in this paper.


# The architecture of the proposed model, UBNet



# Dependencies and Environment
Dependencies can be installed via anaconda.
```
python>=3.7
pytorch==1.9.0
torchvision==0.10.0
numpy=1.18.5
adamp=0.3.0
opencv-python=4.5.3.56
cudatoolkit=11.1.74
```

# Dataset Preparation

## CelebA-HQ
Download the CelebA-HQ and CelebA in the links as below.
- https://github.com/switchablenorms/CelebAMask-HQ
- https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (In-The-Wild Images)


We expect the directory sturcture to be the following
```
dataset/
  celebA/
    celebA/
      img_celeba/     # CelebA images in wild
    celebA-HQ
      CelebA-HQ-img/  # CelebA-HQ images
```

# How to run

## Pretrained weights
Download the weights in https://drive.google.com/drive/folders/1_Dkr4CAPxWHbkOU7PIV3gbg9-oqotWRI?usp=sharing

## CelebA-HQ

### Training
base model
```
python base_model/main_base.py -e celebA_train --imagenet_pretrain --data_dir dataset --save_dir exp --data CelebA-HQ --is_train --model vgg11 --batch_size=32 --max_step=20 --lr=0.0001 --cuda --gpu=0 --lr_scheduler step --lr_decay_period=10
```

UBNet
```
python main.py -e celebA_ubnet_train --is_train --ubnet --cuda --checkpoint exp/celebA_train/checkpoint_step_19.pth --data CelebA-HQ --data_dir dataset --save_dir exp --lr=0.0001 --max_step=20 --gpu=0 --batch_size=32 --model vgg11 --lr_scheduler step --lr_decay_period=10
```
### Evaluation
| Method    	| Base Model   	| HEX          	| Rebias       	| LfF          |**UBNet**       	|
|-----------	|--------------	|--------------	|--------------	|--------------|--------------|
| ACC(UB1)  	| 99.38(±0.31) 	| 92.50(±0.67) 	| 99.05(±0.13) 	|92.25(±4.61)  |**99.18(±0.18)** 	|
| ACC(UB2)  	| 51.22(±1.73) 	| 50.85(±0.37) 	| 55.57(±1.43) 	|56.70(±6.69)  | **58.22(±0.64)** |
| ACC(test) 	| 75.30(±0.93) 	| 71.68(±0.50) 	| 77.31(±0.71) 	|74.98(±4.16)  | **78.70(±0.24)** |

Note that we have reported the average of 3 evaluations. The uploaded weight is from one of the 3 experiments; ACC(EB1) 99.29%, ACC(EB2) 58.74%, and hence ACC(Test) 79.02%. 

```
python main.py -e celebA_ubnet_test --ubnet --cuda --checkpoint_orth weights/celebA_ubnet/celeba_ubnet.pth --checkpoint weights/celebA_baseline/celeba_base.pth --data CelebA-HQ --data_dir dataset --save_dir exp --gpu=0 --batch_size=32 --model vgg11
```



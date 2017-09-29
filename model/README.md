# Segmenting WSI images using CNNs
This folder contains the model files including pre-trained encoder. Our encoder-decoder network that incorporates input-aware residual convolutional units, dense connections between encoder and decoder, and multiple decoding paths is contained in **Model_MRIADCMD.lua** file. Please read our paper for more details.

**Paper:** [Learning to Segment Breast Biopsy Whole Slide Images](https://arxiv.org/pdf/1709.02554.pdf)


## Download the pre-trained encoder
Please download the ResNet-18 pretrained on the ImageNet dataset from the below link
```
https://github.com/facebook/fb.resnet.torch/tree/master/pretrained
```

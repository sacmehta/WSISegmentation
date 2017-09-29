# WSI Segmentation using Deep Convolutional Neural Networks
This repository contains the source code of our paper [Learning to Segment Breast Biopsy Whole Slide Images](https://arxiv.org/pdf/1709.02554.pdf)

## Dataset
Please keep your dataset in data folder. In our experiments, we followed the following structure

-- data
---- mel
---- train.txt
---- val.txt
------ trainrgb
------ trainannot
------ testrgb
------ testannot

The text files (train.txt and val.txt) contains the names of training and validation files. Each of these files stores the image file location in the following format
```
/trainrgb/<image_file1>.png,/trainrgb/<image_file1>.png
/trainrgb/<image_file2>.png,/trainrgb/<image_file2>.png
```

## Training Semantic Segmentation Model
You can train you model by executing the following command:
  ```
      CUDA_VISIBLE_DEVICES=0 th main.lua
  ```
If you wish to modify the parameters, you can pass command line arguments, as below. See **opts.lua** for mode command-line options
```
CUDA_VISIBLE_DEVICES=0 th main.lua -lr 0.0001
```
The above command will run the code with an initial learning rate of 0.0001.

## License
This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).


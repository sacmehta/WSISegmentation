# WSI Segmentation using Deep Convolutional Neural Networks
This repository contains the source code of our paper [Learning to Segment Breast Biopsy Whole Slide Images](https://arxiv.org/pdf/1709.02554.pdf)

If you are using this source code, please cite our following paper:
```
@inproceedings{mehta2017learning,
  title={Learning to Segment Breast Biopsy Whole Slide Images},
  author={Mehta, Sachin and Mercan, Ezgi and Bartlett, Jamen and Weaver, Donald L and Elmore, Joann G and Shapiro, Linda G},
  booktitle={Applications of Computer Vision (WACV), 2018 IEEE Winter Conference on},
  year={2018},
  organization={IEEE}
}

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

## NOTE
The source code is adapted from [MSRSegNet](https://github.com/sacmehta/MSRSegNet)

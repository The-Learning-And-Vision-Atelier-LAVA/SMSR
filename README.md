# SMSR
Reposity for "Learning Sparse Masks for Efficient Image Super-Resolution"

[[arXiv]](https://arxiv.org/abs/2006.09603)


## Highlights

- Locate and skip redundant computation in SR networks at a fine-grained level for efficient inference.
- Maintain state-of-the-art performance with significant FLOPs reduction and a speedup on mobile devices.
- Efficient implementation of sparse convolution based on ***original Pytorch APIs*** for easier migration and deployment.


## Network Architecture

<img width="750" src="https://github.com/LongguangWang/SMSR/blob/master/Figs/overview.png"/></div>


<img width="650" src="https://github.com/LongguangWang/SMSR/blob/master/Figs/sparse conv.png"/></div>


## Implementation of Sparse Convolution
For easier migration and deployment, we use an efficient implementation of sparse convolution based on original Pytorch APIs rather than the commonly applied CUDA-based implementation. Specifically, sparse features are first extracted from the input, as shown in the following figure. Then, matrix multiplication is executed to produce the output features.

<img width="850" src="https://github.com/LongguangWang/SMSR/blob/master/Figs/implementation.png"/></div>


## Requirements
- Python 3.6
- PyTorch == 1.1.0
- numpy
- skimage
- imageio
- matplotlib
- cv2


## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train
```bash
python main.py --model SMSR --save SMSR_X2 --scale 2 --patch_size 96 --batch_size 16
```

## Test
### Prepare test data 
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in `testset/benchmark` following the example of `testset/benchmark/Set5`.


### Demo
```bash
python main.py --dir_data testsets --data_test Set5 --scale 2 --model SMSR --save SMSR_X2 --pre_train experiment/SMSR_X2/model/model_1000.pt --test_only --save_results
```


## Results

<img width="650" src="https://github.com/LongguangWang/SMSR/blob/master/Figs/results.png"/></div>


## Visualization of Sparse Masks

<img width="350" src="https://github.com/LongguangWang/SMSR/blob/master/Figs/visualization.png"/></div>



## Citation
```
@Article{Wang2020Learning,
  author  = {Wang, Longguang and Dong, Xiaoyu and Wang, Yingqian and Ying, Xinyi and Lin, Zaiping and An, Wei and Guo, Yulan},
  title   = {Learning Sparse Masks for Efficient Image Super-Resolution},
  journal = {arXiv},
  year    = {2020},
}
```

## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing the codes.


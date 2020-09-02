# SMSR
Reposity for "Learning Sparse Masks for Efficient Image Super-Resolution"

[[arXiv]](https://arxiv.org/abs/2006.09603)

- Locate and skip redundant computation in SR networks at a fine-grained level for efficient inference.
- Maintain state-of-the-art performance with significant FLOPs reduction and a speedup on mobile devices.
- Efficient implementation of sparse convolution based on ***original Pytorch APIs*** for easier migration and deployment.

**Code will be released soon. Stay tuned.**


## Network Architecture

<img width="750" src="https://github.com/LongguangWang/SMSR/blob/master/Figs/overview.png"/></div>


<img width="650" src="https://github.com/LongguangWang/SMSR/blob/master/Figs/sparse conv.png"/></div>


## Implementation of Sparse Convolution
For easier migration and deployment, we use an efficient implementation of sparse convolution based on original Pytorch APIs rather than the commonly applied CUDA-based implementation. Specifically, sparse features are first extracted from the input, as shown in the following figure. Then, matrix multiplication is executed to produce the output features.

<img width="850" src="https://github.com/LongguangWang/SMSR/blob/master/Figs/implementation.png"/></div>

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

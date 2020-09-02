# SMSR
Reposity for "Learning Sparse Masks for Efficient Super-Resolution"

[[arXiv]](https://arxiv.org/abs/2006.09603)

- Locate and skip redundant computation in SR networks at a fine-grained level for efficient inference.
- Maintain state-of-the-art performance with significant FLOPs reduction and a speedup on mobile devices.
- Efficient implementation of sparse convolution based on **original Pytorch APIs** for easier migration and deployment.

## Network Architecture

<img width="550" src="https://github.com/LongguangWang/PAM/blob/master/Figs/overview.png"/></div>


<img width="550" src="https://github.com/LongguangWang/PAM/blob/master/Figs/sparse conv.png"/></div>


## Results

<img width="550" src="https://github.com/LongguangWang/PAM/blob/master/Figs/results.png"/></div>


## Visualization of Sparse Masks

<img width="550" src="https://github.com/LongguangWang/PAM/blob/master/Figs/visualization.png"/></div>

Codes will be released soon. Stay tuned.


## Citation
```
@Article{Wang2020Learninga,
  author  = {Wang, Longguang and Dong, Xiaoyu and Wang, Yingqian and Ying, Xinyi and Lin, Zaiping and An, Wei and Guo, Yulan},
  title   = {Learning Sparse Masks for Efficient Image Super-Resolution},
  journal = {arXiv},
  year    = {2020},
}
```

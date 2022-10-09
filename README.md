# Tri-Attention <br> (Official PyTorch Implementation)

[![arXiv](https://img.shields.io/badge/arXiv-2208.00219-b31b1b.svg)](https://arxiv.org/abs/2208.00219)

-------

This repository is the official PyTorch implementation of the
**** 2022 paper "[Tri-Attention: Explicit Context-Aware Attention Mechanism for Natural Language Processing](https://doi.org/*****)" by _Rui Yu, Yifeng Li, Wenpeng Lu and Longbing Cao_. 

<b> [ Important Notice ] </b> &nbsp;&nbsp; Tri-Attention first appeared as a tech report on arXiv.org (https://arxiv.org/abs/*****) in 2022. Since its release, we have made substantial improvements to the original version. Please kindly be advised to refer to the latest version of the paper.


-------
&nbsp;
## Brief Introduction

The proposed Tri-Attention mechanism expands the standard two-dimensional attention framework to explicitly involve and couple contextual information with query and key, hence the attention weights more sufficiently capture context-aware sequence interactions. To the best of our knowledge, this is the first work on explicitly involving context (contextual features) and learning query-key-context interaction-based attention between sequences.
Tri-attention takes a general three-dimensional tensor framework, which can be instantiated into different implementations and applied to various tasks. We illustrate four variants by expanding the additive, dot-product, scaled dot-product and trilinear operations on query, key and context using tensor algebra for calculating Tri-Attention.

Please check [our paper](https://doi.org/****) or [its preprint version](https://arxiv.org/abs/****) for more details.


-------
&nbsp;

## Dialogue

### Installation

#### Pre-Requisites
You must have NVIDIA GPUs to run the codes.

The implementation codes are developed and tested with the following environment setups:
- numpy==1.19.5
- setproctitle==1.2.2
- torch==1.8.0.dev20210113+cu110
- torchvision==0.9.0.dev20210113+cu110
- tqdm==4.56.2
- transformers==2.8.0

We recommend using the exact setups above. However, other environments (Linux, Python>=3.7, CUDA>=9.2, GCC>=5.4, PyTorch>=1.5.1, TorchVision>=0.6.1) should also work properly.

#### Code Installation

This code is implemented using PyTorch v1.8.0, and provides out of the box support with CUDA 11.2
Anaconda is the recommended to set up this codebase.
```
# https://pytorch.org
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

#### Preparing Data and Checkpoints
-------------

#### Post-trained and fine-tuned Checkpoints

We provide following post-trained and fine-tuned checkpoints. 

- [fine-grained post-trained checkpoint for ubuntu benchmark dataset][3]
- [fine-tuned checkpoint for ubuntu benchmark dataset ][4]


#### Data pkl for Fine-tuning (Response Selection)
We used the following data for post-training and fine-tuning
- [fine-grained post-training dataset and fine-tuning dataset for ubuntu benchmark ][5]


Original version for ubuntu dataset is availble in [Ubuntu Corpus V1][6].

----------

### Usage

##### Making Data for post-training and fine-tuning  

```
Data_processing.py
```


### Post-training Example
```shell
python -u FPT/ubuntu_final.py --num_train_epochs 25
```

### Fine-tuning Example

###### Taining 
```shell
To train the model, set `--is_training`
python -u Fine-Tuning/Response_selection.py --task ubuntu --is_training
```
###### Testing
```shell
python -u Fine-Tuning/Response_selection.py --task ubuntu
```






## RC

### Installation

#### Pre-Requisites
You must have NVIDIA GPUs to run the codes.

The implementation codes are developed and tested with the following environment setups:
- numpy==1.19.5
- setproctitle==1.2.2
- torch==1.8.0.dev20210113+cu110
- torchvision==0.9.0.dev20210113+cu110
- tqdm==4.56.2
- transformers==2.8.0

We recommend using the exact setups above. However, other environments (Linux, Python>=3.7, CUDA>=9.2, GCC>=5.4, PyTorch>=1.5.1, TorchVision>=0.6.1) should also work properly.

#### Code Installation

This code is implemented using PyTorch v1.8.0, and provides out of the box support with CUDA 11.2
Anaconda is the recommended to set up this codebase.
```
# https://pytorch.org
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

#### Preparing Data and Checkpoints
-------------

#### Post-trained and fine-tuned Checkpoints

We provide following post-trained and fine-tuned checkpoints. 

- [fine-grained post-trained checkpoint for ubuntu benchmark dataset][3]
- [fine-tuned checkpoint for ubuntu benchmark dataset ][4]


#### Data pkl for Fine-tuning (Response Selection)
We used the following data for post-training and fine-tuning
- [fine-grained post-training dataset and fine-tuning dataset for ubuntu benchmark ][5]


Original version for ubuntu dataset is availble in [Ubuntu Corpus V1][6].

----------

### Usage

##### Making Data for post-training and fine-tuning  

```
Data_processing.py
```


### Post-training Example
```shell
python -u FPT/ubuntu_final.py --num_train_epochs 25
```

### Fine-tuning Example

###### Taining 
```shell
To train the model, set `--is_training`
python -u Fine-Tuning/Response_selection.py --task ubuntu --is_training
```
###### Testing
```shell
python -u Fine-Tuning/Response_selection.py --task ubuntu
```



## SPM

### Installation

#### Pre-Requisites
You must have NVIDIA GPUs to run the codes.

The implementation codes are developed and tested with the following environment setups:
- numpy==1.19.5
- setproctitle==1.2.2
- torch==1.8.0.dev20210113+cu110
- torchvision==0.9.0.dev20210113+cu110
- tqdm==4.56.2
- transformers==2.8.0

We recommend using the exact setups above. However, other environments (Linux, Python>=3.7, CUDA>=9.2, GCC>=5.4, PyTorch>=1.5.1, TorchVision>=0.6.1) should also work properly.

#### Code Installation

This code is implemented using PyTorch v1.8.0, and provides out of the box support with CUDA 11.2
Anaconda is the recommended to set up this codebase.
```
# https://pytorch.org
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

#### Preparing Data and Checkpoints
-------------

#### Post-trained and fine-tuned Checkpoints

We provide following post-trained and fine-tuned checkpoints. 

- [fine-grained post-trained checkpoint for ubuntu benchmark dataset][3]
- [fine-tuned checkpoint for ubuntu benchmark dataset ][4]


#### Data pkl for Fine-tuning (Response Selection)
We used the following data for post-training and fine-tuning
- [fine-grained post-training dataset and fine-tuning dataset for ubuntu benchmark ][5]


Original version for ubuntu dataset is availble in [Ubuntu Corpus V1][6].

----------

### Usage

##### Making Data for post-training and fine-tuning  

```
Data_processing.py
```


### Post-training Example
```shell
python -u FPT/ubuntu_final.py --num_train_epochs 25
```

### Fine-tuning Example

###### Taining 
```shell
To train the model, set `--is_training`
python -u Fine-Tuning/Response_selection.py --task ubuntu --is_training
```
###### Testing
```shell
python -u Fine-Tuning/Response_selection.py --task ubuntu
```



----------

&nbsp;
## Citation

If you find Meta-DETR useful or inspiring, please consider citing:

```bibtex
@article{Meta-DETR-2022,
  author={Zhang, Gongjie and Luo, Zhipeng and Cui, Kaiwen and Lu, Shijian and Xing, Eric P.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={{Meta-DETR}: Image-Level Few-Shot Detection with Inter-Class Correlation Exploitation}, 
  year={2022},
  doi={10.1109/TPAMI.2022.3195735},
}
```

----------
&nbsp;
## Acknowledgement

Our proposed Meta-DETR is heavily inspired by many outstanding prior works, including [DETR](https://github.com/facebookresearch/detr) and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Thank the authors of above projects for open-sourcing their implementation codes!



[1]: https://github.com/huggingface/transformers
[2]: https://github.com/taesunwhang/BERT-ResSel
[3]: https://drive.google.com/file/d/1-4E0eEjyp7n_F75TEh7OKrpYPK4GLNoE/view?usp=sharing
[4]: https://drive.google.com/file/d/1n2zigNDiIArWtsiV9iUQLwfSBgtNn7ws/view?usp=sharing
[5]: https://drive.google.com/file/d/16Rv8rSRneq7gfPRkpFZseNYfswuoqI4-/view?usp=sharing
[6]: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip
[7]: https://github.com/MarkWuNLP/MultiTurnResponseSelection
[8]: https://github.com/cooelf/DeepUtteranceAggregation

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


### Post-training Examples

##### (Ubuntu Corpus V1, Douban Corpus, E-commerce Corpus)

```shell
python -u FPT/ubuntu_final.py --num_train_epochs 25
```

### Fine-tuning Examples

##### (Ubuntu Corpus V1, Douban Corpus, E-commerce Corpus)

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



## SPM





## Installation

### Pre-Requisites
You must have NVIDIA GPUs to run the codes.

The implementation codes are developed and tested with the following environment setups:
- numpy==1.19.5
- setproctitle==1.2.2
- torch==1.8.0.dev20210113+cu110
- torchvision==0.9.0.dev20210113+cu110
- tqdm==4.56.2
- transformers==2.8.0

We recommend using the exact setups above. However, other environments (Linux, Python>=3.7, CUDA>=9.2, GCC>=5.4, PyTorch>=1.5.1, TorchVision>=0.6.1) should also work properly.

&nbsp;

### Code Installation

This code is implemented using PyTorch v1.8.0, and provides out of the box support with CUDA 11.2
Anaconda is the recommended to set up this codebase.
```
# https://pytorch.org
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

&nbsp;

### Data Preparation

#### Ubuntu Corpus V1 for retrieval-based dialogues

Please download [Ubuntu Corpus V1 dataset](https://cocodataset.org/) and organize them as following:

```
Dialogue/
└── ubuntu_data/
```

#### RACE for machine reading comprehension

Please download [RACE](https://cocodataset.org/) and organize them as following:

```
RC/
└── RACE/
    └── train/
    └── dev/
    └── test/
```

#### LCQMC for sentence semantic matching

Please download [LCQMC](https://cocodataset.org/) and organize them as following:

```
SPM/
└── LCQMC/
    └── train/
    └── dev/
    └── test/
```

----------
&nbsp;

## Usage

### Reproducing Paper Results

All scripts to reproduce results reported in [our T-PAMI paper](https://doi.org/10.1109/TPAMI.2022.3195735)
are stored in [`./scripts`](scripts). The arguments are pretty easy and straightforward to understand. 

Taking MS-COCO as an example, run the following commands to reproduce paper results:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./scripts/run_experiments_coco.sh
```


### To Perform _**Base Training**_

We take MS-COCO as an example. First, create `basetrain.sh` and copy the following commands into it.
```bash
EXP_DIR=exps/coco
BASE_TRAIN_DIR=${EXP_DIR}/base_train
mkdir exps
mkdir ${EXP_DIR}
mkdir ${BASE_TRAIN_DIR}

python -u main.py \
    --dataset_file coco_base \
    --backbone resnet101 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 4 \
    --category_codes_cls_loss \
    --epoch 25 \
    --lr_drop_milestones 20 \
    --save_every_epoch 5 \
    --eval_every_epoch 5 \
    --output_dir ${BASE_TRAIN_DIR} \
2>&1 | tee ${BASE_TRAIN_DIR}/log.txt
```
Then, run the commands below to start base training.
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8  ./basetrain.sh
```

### To Perform _**Few-Shot Finetuning**_
We take MS-COCO as an example. First, create `fsfinetune.sh` and copy the following commands into it.
```bash
EXP_DIR=exps/coco
BASE_TRAIN_DIR=${EXP_DIR}/base_train
mkdir exps
mkdir ${EXP_DIR}
mkdir ${BASE_TRAIN_DIR}

fewshot_seed=01
num_shot=10
epoch=500
lr_drop1=300
lr_drop2=450
FS_FT_DIR=${EXP_DIR}/seed${fewshot_seed}_${num_shot}shot
mkdir ${FS_FT_DIR}

python -u main.py \
    --dataset_file coco_base \
    --backbone resnet101 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 2 \
    --category_codes_cls_loss \
    --resume ${BASE_TRAIN_DIR}/checkpoint.pth \
    --fewshot_finetune \
    --fewshot_seed ${fewshot_seed} \
    --num_shots ${num_shot} \
    --epoch ${epoch} \
    --lr_drop_milestones ${lr_drop1} ${lr_drop2} \
    --warmup_epochs 50 \
    --save_every_epoch ${epoch} \
    --eval_every_epoch ${epoch} \
    --output_dir ${FS_FT_DIR} \
2>&1 | tee ${FS_FT_DIR}/log.txt
```
Note that you need to add `--fewshot_finetune` to indicate that the training and inference should be conducted on few-shot setups. You also need to specify the number of shots, few-shot random seed, training epoch setups, and the checkpoint file path after base training.
Then, run the commands below to start few-shot finetuning. After finetuning, the program will automatically perform inference on novel classes.
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8  ./fsfinetune.sh
```


### To Perform _**Only Inference**_ After Few-Shot Finetuning

We take MS-COCO as an example. Simply run:
```bash
python -u main.py \
--dataset_file coco_base \
--backbone resnet101 \
--num_feature_levels 1 \
--enc_layers 6 \
--dec_layers 6 \
--hidden_dim 256 \
--num_queries 300 \
--batch_size 2 \
--category_codes_cls_loss \
--resume path/to/checkpoint.pth/generated/by/few-shot-fintuning \
--fewshot_finetune \
--fewshot_seed ${fewshot_seed} \
--num_shots ${num_shot} \
--eval \
2>&1 | tee ./log_inference.txt
```
Note that user should set `--eval` and `--resume path/to/checkpoint.pth/generated/by/few-shot-fintuning` correctly.



-----------
&nbsp;
## Pre-Trained Model Weights

We provide trained model weights after __the base training stage__ for users to finetune.

*All pre-trained model weights are stored in __Google Drive__.*

- __MS-COCO__ after base training:&nbsp;&nbsp; click [here](https://drive.google.com/file/d/19tfI_XNZolDId_G5s45YTgcFKt8Ji7c8/view?usp=sharing) to download.

- __Pascal VOC Split 1__ after base training:&nbsp;&nbsp; click [here](https://drive.google.com/file/d/1e3xHnVVsS3JFNGTfh51xjUPPZVtwTGOq/view?usp=sharing) to download.

- __Pascal VOC Split 2__ after base training:&nbsp;&nbsp; click [here](https://drive.google.com/file/d/1SMOQP-ZKnuIrg3R32a-6FYtA3zkWeNF2/view?usp=sharing) to download.

- __Pascal VOC Split 3__ after base training:&nbsp;&nbsp; click [here](https://drive.google.com/file/d/1EJ6uP3yAequS5Wl3gEDtyxKx8ZfgPhAi/view?usp=sharing) to download.



----------

&nbsp;
## License

The implementation codes of Meta-DETR are released under the MIT license.

Please see the [LICENSE](LICENSE) file for more information.

However, prior works' licenses also apply. It is the users' responsibility to ensure compliance with all license requirements.


------------

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

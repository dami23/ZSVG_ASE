This repository provides the official PyTorch implementation of:

**Exploring Attribute-Driven Specificity for Zero-Shot Visual Grounding**

## 1. Dataset Preparation

Please download and prepare the required datasets before training or evaluation.

The project supports the following visual grounding datasets:
- RefCOCO
- RefCOCO+
- RefCOCOg
- Flickr30K Entities
- Flickr-Split-0/1
- VG-2B/2UB/3B/3UB
  
Please organize the datasets according to the directory structure expected by the configuration files in this repository, and download the required csv files from https://github.com/TheShadow29/zsgnet-pytorch.

project_root/
├── codes/
├── datasets/
│   ├── refcoco/
│   ├── refcoco+/
│   ├── refcocog/
│   ├── flickr30k/
│   └── vg/
├── checkpoints/
└── README.md

## 2. Model Training

### 2.1 Select GPUs
Specify the GPUs to be used:
export CUDA_VISIBLE_DEVICES=0,1

### 2.2 Train the Model
python codes/model_train.py refcoco_try

Here, refcoco_try denotes the experiment or configuration name. Replace it with the corresponding configuration name for other datasets or experimental settings.

### 2.3 Evaluate the Model
python codes/model_train.py refcoco_try --resume=True --only_test=True

## 3. Pretrained Models
The pretrained models can be downloaded from Google Drive:
After downloading, place the checkpoint files in the corresponding checkpoint directory, for example:
checkpoints/
└── refcoco_try/
    └── checkpoint_best.pth

## 4. Acknowledgements
This implementation is developed using PyTorch. We thank the authors and maintainers of the datasets and related open-source projects used in this work.

This project is the pyTorch implementation of "Exploring Attribute-driven Specificity for Zero-Shot Visual Grounding".


export CUDA_VISIBLE_DEVICES=0,1

train
python codes/model_train.py 'refcoco_try'

valuation
python codes/model_train.py 'refcoco_try' --resume=True --only_test=True

The pretrained models can be downloaded from google.com

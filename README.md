# Efficient end-to-end learning for quantizable representations
This repository has the source code for the paper "Efficient end-to-end learning for quantizable representations"(ICML18).

## Citing this work
```
@inproceedings{jeongICML18,
    title={Efficient end-to-end learning for quantizable representations},
    author={Jeong, Yeonwoo and Song, Hyun Oh},
    booktitle={arXiv preprint arXiv:1805.05809},
    year={2018}
    }
```

## Installation
* Python3.5
* Deep learning frame work : Tensorflow1.4 gpu
Check [https://github.com/tensorflow/tensorflow/tree/r1.4](https://github.com/tensorflow/tensorflow/tree/r1.4)
* Ortools(6.6.4656)
Check [https://developers.google.com/optimization/introduction/download](https://developers.google.com/optimization/introduction/download)

## Prerequisites
1. Make Directory for data and experiment
```
cd RROOT
mkdir dataset deep_hash_table_processed deep_hash_table_exp_results
mkdir dataset/Imagenet32
``` 
2. Change path in config/path.py
```
RROOT = '(user enter path)'
EXP_PATH = RROOT+'deep_hash_table_exp_results/'
#=============CIFAR100============================#
CIFAR100PATH = RROOT+'dataset/cifar-100-python/'
CIFARPROCESSED = RROOT+'deep_hash_table_processed/cifar_processed/'
#==========================Imagenet32===============================#
IMAGENET32PATH = RROOT+'dataset/Imagenet32/'
IMAGENET32PROCESSED = RROOT+'deep_hash_table_processed/Imagenet32_processed/'
```
3. Download and unzip dataset Cifar-100 and Downsampled imagenet(32x32)
```
cd RROOT/dataset
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar zxvf cifar-100-python.tar.gz

cd RROOT/dataset/Imagenet32
wget http://www.image-net.org/image/downsample/Imagenet32_train.zip
wget http://www.image-net.org/image/downsample/Imagenet32_val.zip
unzip Imagenet32_train.zip
unzip Imagenet32_val.zip
```
## Processing Data
```
cd process
python cifar_process.py
python imagenet32_process.py 
```

## Training Procedure
1. Training metric
2. Training hash codes

## Evaluation

## Ortools

## License 
MIT License


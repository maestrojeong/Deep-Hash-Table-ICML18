import sys
sys.path.append("../../configs")
#../../configs
from path import EXP_PATH

import numpy as np

DECAY_PARAMS_DICT =\
    {
    'stair' : 
        {
            'type-2': {'initial_lr' : 1e-7, 'decay_steps' : 40000, 'decay_rate' : 0.1},
            'type-1': {'initial_lr' : 1e-6, 'decay_steps' : 40000, 'decay_rate' : 0.1},
            'type0': {'initial_lr' : 1e-5, 'decay_steps' : 40000, 'decay_rate' : 0.1},
            'type1': {'initial_lr' : 1e-4, 'decay_steps' : 40000, 'decay_rate' : 0.1},
            'type2' : {'initial_lr' : 3e-4, 'decay_steps' : 40000, 'decay_rate' : 0.1},
            'type3' : {'initial_lr' : 1e-3, 'decay_steps' : 40000, 'decay_rate' : 0.1},
            'type4' : {'initial_lr' : 3e-3, 'decay_steps' : 40000, 'decay_rate' : 0.1},
            'type5' : {'initial_lr' : 1e-2, 'decay_steps' : 40000, 'decay_rate' : 0.1},
            'type6' : {'initial_lr' : 3e-2, 'decay_steps' : 40000, 'decay_rate' : 0.1},
            'type7' : {'initial_lr' : 1e-1, 'decay_steps' : 40000, 'decay_rate' : 0.1}
        },
   'piecewise' :
        {
            'type1' : {'boundaries' : [10000, 100000, 100000], 'values' : [1e-6, 1e-4, 1e-5, 1e-6]},
            'type2' : {'boundaries' : [10000, 100000, 100000], 'values' : [1e-6, 5e-4, 5e-5, 5e-6]},
            'type3' : {'boundaries' : [10000, 100000, 100000], 'values' : [1e-6, 1e-3, 1e-4, 1e-5]},
            'type4' : {'boundaries' : [10000, 100000, 100000], 'values' : [1e-6, 1e-2, 1e-3, 1e-4]},
            'type5' : {'boundaries' : [10000, 100000, 100000], 'values' : [1e-6, 1e-1, 1e-2, 1e-3]},
        }
    }
K_SET = [1,4,16]
RESULT_DIR = EXP_PATH + 'imagenet_exps/'
#========================PARAM============================#
DATASET= 'cifar'
GPU_ID = 0
BATCH_SIZE = 128
NSCLASS = 8
# model
EMBED_M= 64
CONV_NAME = 'conv2'
PRETRAIN = False # Load convolution netowrk or not 
# metric loss
LOSS_TYPE = 'npair'
MARGIN_ALPHA = 0.5
LAMBDA = 0.002 # regularization for npair
# learning
DECAY_TYPE = 'stair'
DECAY_PARAM_TYPE = 'type3' 
# hash
BUCKET_D = 64
ACTIVATE_K = 1
HASH_METRIC_SHAPE = 'par' # 'seq'
# hash loss
HASH_LOSS_TYPE = 'npair'
HASH_DISTANCE_TYPE = 'hash1'
HASH_LAMBDA = 0.003
HASH_MARGIN_ALPHA = 0.1
# hash learning
HASH_DECAY_TYPE = 'stair'
HASH_DECAY_PARAM_TYPE = 'type1' 
# minimum cost flow
PAIRWISE_LAMBDA = 0.01 

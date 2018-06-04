import sys
sys.path.append("../../configs")
#../../configs
from path import EXP_PATH

import numpy as np

DECAY_PARAMS_DICT =\
    {
    'stair' : 
        {
            512 :{
                'a1': {'initial_lr' : 1e-5, 'decay_steps' : 30000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-4, 'decay_steps' : 30000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-3, 'decay_steps' : 30000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-3, 'decay_steps' : 30000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-2, 'decay_steps' : 30000, 'decay_rate' : 0.3},
                'b1': {'initial_lr' : 1e-5, 'decay_steps' : 40000, 'decay_rate' : 0.3},
                'b2' : {'initial_lr' : 3e-4, 'decay_steps' : 40000, 'decay_rate' : 0.3},
                'b3' : {'initial_lr' : 1e-3, 'decay_steps' : 40000, 'decay_rate' : 0.3},
                'b4' : {'initial_lr' : 3e-3, 'decay_steps' : 40000, 'decay_rate' : 0.3},
                'b5' : {'initial_lr' : 1e-2, 'decay_steps' : 40000, 'decay_rate' : 0.3}
                },
            256 :{
                'a1': {'initial_lr' : 1e-5, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-4, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-3, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-3, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-2, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'b1': {'initial_lr' : 1e-5, 'decay_steps' : 60000, 'decay_rate' : 0.3},
                'b2' : {'initial_lr' : 3e-4, 'decay_steps' : 60000, 'decay_rate' : 0.3},
                'b3' : {'initial_lr' : 1e-3, 'decay_steps' : 60000, 'decay_rate' : 0.3},
                'b4' : {'initial_lr' : 3e-3, 'decay_steps' : 60000, 'decay_rate' : 0.3},
                'b5' : {'initial_lr' : 1e-2, 'decay_steps' : 60000, 'decay_rate' : 0.3}
                }
        },
   'piecewise' :
        {
            1024 : {
                'a1' : {'boundaries' : [10000, 20000], 'values' : [1e-4, 3e-5, 1e-5]},
                'a2' : {'boundaries' : [10000, 20000], 'values' : [3e-4, 1e-4, 3e-5]},
                'a3' : {'boundaries' : [10000, 20000], 'values' : [1e-3, 3e-4, 1e-4]},
                'a4' : {'boundaries' : [10000, 20000], 'values' : [3e-3, 1e-3, 3e-4]},
                'a5' : {'boundaries' : [10000, 20000], 'values' : [1e-2, 3e-3, 1e-3]}
                 },
            512 : {
                'a1' : {'boundaries' : [20000, 40000], 'values' : [1e-4, 3e-5, 1e-5]},
                'a2' : {'boundaries' : [20000, 40000], 'values' : [3e-4, 1e-4, 3e-5]},
                'a3' : {'boundaries' : [20000, 40000], 'values' : [1e-3, 3e-4, 1e-4]},
                'a4' : {'boundaries' : [20000, 40000], 'values' : [3e-3, 1e-3, 3e-4]},
                'a5' : {'boundaries' : [20000, 40000], 'values' : [1e-2, 3e-3, 1e-3]},
                'b1' : {'boundaries' : [40000, 70000], 'values' : [1e-4, 3e-5, 1e-5]},
                'b2' : {'boundaries' : [40000, 70000], 'values' : [3e-4, 1e-4, 3e-5]},
                'b3' : {'boundaries' : [40000, 70000], 'values' : [1e-3, 3e-4, 1e-4]},
                'b4' : {'boundaries' : [40000, 70000], 'values' : [3e-3, 1e-3, 3e-4]},
                'b5' : {'boundaries' : [40000, 70000], 'values' : [1e-2, 3e-3, 1e-3]}
                 },
        }
    }
K_SET = [1,4,16]
RESULT_DIR = EXP_PATH+"imagenet_exps/"
#========================PARAM============================#
DATASET= 'imgnet32'
GPU_ID = 0
BATCH_SIZE = 1024
NSCLASS = 512
EPOCH = 100
# model
EMBED_M=512 
CONV_NAME = 'conv1'
LOSS_TYPE = 'npair'
# hash
BUCKET_D = 512 # npair 512 triplet 256
ACTIVATE_K = 1
# hash loss
HASH_LOSS_TYPE = 'npair'
HASH_DISTANCE_TYPE = 'h1'
HASH_LAMBDA = 0.003
HASH_MARGIN_ALPHA = 1.0
# hash learning
HASH_DECAY_TYPE = 'piecewise'
HASH_DECAY_PARAM_TYPE = 'b2' 
# minimum cost flow
PAIRWISE_LAMBDA = 10.0 # npair 10.0 triplet 1.0

import sys
sys.path.append("../../configs")
#../../configs
from path import EXP_PATH

import numpy as np

DECAY_PARAMS_DICT =\
    {
    'stair' : 
        {
            256 :{
                'a1': {'initial_lr' : 1e-4, 'decay_steps' : 200000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-4, 'decay_steps' : 200000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-3, 'decay_steps' : 200000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-3, 'decay_steps' : 200000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-2, 'decay_steps' : 200000, 'decay_rate' : 0.3}
                },
            512 :{
                'a1': {'initial_lr' : 1e-4, 'decay_steps' : 150000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-4, 'decay_steps' : 150000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-3, 'decay_steps' : 150000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-3, 'decay_steps' : 150000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-2, 'decay_steps' : 150000, 'decay_rate' : 0.3}
                }
        }
    }
K_SET = [1,4,16]
RESULT_DIR = EXP_PATH+"imagenet_exps/"
#========================PARAM============================#
DATASET= 'imgnet32'
GPU_ID = 0
BATCH_SIZE = 512
EPOCH = 200
NSCLASS = 256
# model
EMBED_M = 512
CONV_NAME = 'conv1' 
# metric loss
LOSS_TYPE = 'npair'
MARGIN_ALPHA = 1.0
LAMBDA = 0.002 # regularization for npair
# learning
DECAY_TYPE = 'stair'
DECAY_PARAM_TYPE = 'a3' 


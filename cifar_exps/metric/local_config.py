import sys
sys.path.append("../../configs")
#../../configs
from path import EXP_PATH
import numpy as np
DECAY_PARAMS_DICT =\
    {
    'stair' : 
        {
            128 :{
                'a1': {'initial_lr' : 1e-5, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-4, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-3, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-3, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-2, 'decay_steps' : 50000, 'decay_rate' : 0.3}
                }
        },
   'piecewise' :
        {
            128 : {
                'a1' : {'boundaries' : [10000, 20000], 'values' : [1e-4, 3e-5, 1e-5]},
                'a2' : {'boundaries' : [10000, 20000], 'values' : [3e-4, 1e-4, 3e-5]},
                'a3' : {'boundaries' : [10000, 20000], 'values' : [1e-3, 3e-4, 1e-4]},
                'a4' : {'boundaries' : [10000, 20000], 'values' : [3e-3, 1e-3, 3e-4]},
                'a5' : {'boundaries' : [10000, 20000], 'values' : [1e-2, 3e-3, 1e-3]},
                'b1' : {'boundaries' : [20000, 35000], 'values' : [1e-4, 3e-5, 1e-5]},
                'b2' : {'boundaries' : [20000, 35000], 'values' : [3e-4, 1e-4, 3e-5]},
                'b3' : {'boundaries' : [20000, 35000], 'values' : [1e-3, 3e-4, 1e-4]},
                'b4' : {'boundaries' : [20000, 35000], 'values' : [3e-3, 1e-3, 3e-4]},
                'b5' : {'boundaries' : [20000, 35000], 'values' : [1e-2, 3e-3, 1e-3]}
                 }
        }
    }
ACTIVATE_K_SET = np.arange(1, 5)
K_SET = [1,4,16]
RESULT_DIR = EXP_PATH+"cifar_exps/"
#========================PARAM============================#
DATASET= 'cifar'
GPU_ID = 0
BATCH_SIZE = 128
EPOCH = 300
NSCLASS = 16
# model
EMBED_M= 64
CONV_NAME = 'conv1'
# metric loss
LOSS_TYPE = 'triplet'
MARGIN_ALPHA = 0.3
LAMBDA = 0.003 # regularization for npair
# learning
DECAY_TYPE = 'stair'
DECAY_PARAM_TYPE = 'a3'


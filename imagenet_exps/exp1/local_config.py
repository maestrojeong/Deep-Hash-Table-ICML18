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

                },
            128 : {
                'a1': {'initial_lr' : 1e-5, 'decay_steps' : 100000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-4, 'decay_steps' : 100000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-3, 'decay_steps' : 100000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-3, 'decay_steps' : 100000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-2, 'decay_steps' : 100000, 'decay_rate' : 0.3},
                'b1': {'initial_lr' : 1e-5, 'decay_steps' : 120000, 'decay_rate' : 0.3},
                'b2' : {'initial_lr' : 3e-4, 'decay_steps' : 120000, 'decay_rate' : 0.3},
                'b3' : {'initial_lr' : 1e-3, 'decay_steps' : 120000, 'decay_rate' : 0.3},
                'b4' : {'initial_lr' : 3e-3, 'decay_steps' : 120000, 'decay_rate' : 0.3},
                'b5' : {'initial_lr' : 1e-2, 'decay_steps' : 120000, 'decay_rate' : 0.3}
                },
            64 : {
                'a1': {'initial_lr' : 1e-5, 'decay_steps' : 200000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-4, 'decay_steps' : 200000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-3, 'decay_steps' : 200000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-3, 'decay_steps' : 200000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-2, 'decay_steps' : 200000, 'decay_rate' : 0.3},
                'b1': {'initial_lr' : 1e-5, 'decay_steps' : 240000, 'decay_rate' : 0.3},
                'b2' : {'initial_lr' : 3e-4, 'decay_steps' : 240000, 'decay_rate' : 0.3},
                'b3' : {'initial_lr' : 1e-3, 'decay_steps' : 240000, 'decay_rate' : 0.3},
                'b4' : {'initial_lr' : 3e-3, 'decay_steps' : 240000, 'decay_rate' : 0.3},
                'b5' : {'initial_lr' : 1e-2, 'decay_steps' : 240000, 'decay_rate' : 0.3}
                },
             32 : {
                'a1': {'initial_lr' : 1e-5, 'decay_steps' : 400000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-4, 'decay_steps' : 400000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-3, 'decay_steps' : 400000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-3, 'decay_steps' : 400000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-2, 'decay_steps' : 400000, 'decay_rate' : 0.3},
                'b1': {'initial_lr' : 1e-5, 'decay_steps' : 480000, 'decay_rate' : 0.3},
                'b2' : {'initial_lr' : 3e-4, 'decay_steps' : 480000, 'decay_rate' : 0.3},
                'b3' : {'initial_lr' : 1e-3, 'decay_steps' : 480000, 'decay_rate' : 0.3},
                'b4' : {'initial_lr' : 3e-3, 'decay_steps' : 480000, 'decay_rate' : 0.3},
                'b5' : {'initial_lr' : 1e-2, 'decay_steps' : 480000, 'decay_rate' : 0.3}
                },
             16 : {
                'a1': {'initial_lr' : 1e-5, 'decay_steps' : 800000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-4, 'decay_steps' : 800000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-3, 'decay_steps' : 800000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-3, 'decay_steps' : 800000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-2, 'decay_steps' : 800000, 'decay_rate' : 0.3},
                'b1': {'initial_lr' : 1e-5, 'decay_steps' : 960000, 'decay_rate' : 0.3},
                'b2' : {'initial_lr' : 3e-4, 'decay_steps' : 960000, 'decay_rate' : 0.3},
                'b3' : {'initial_lr' : 1e-3, 'decay_steps' : 960000, 'decay_rate' : 0.3},
                'b4' : {'initial_lr' : 3e-3, 'decay_steps' : 960000, 'decay_rate' : 0.3},
                'b5' : {'initial_lr' : 1e-2, 'decay_steps' : 960000, 'decay_rate' : 0.3}
                }
        },
   'piecewise' :
        {
            1024 : {
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
            256 : {
                'a1' : {'boundaries' : [30000, 60000], 'values' : [1e-4, 3e-5, 1e-5]},
                'a2' : {'boundaries' : [30000, 60000], 'values' : [3e-4, 1e-4, 3e-5]},
                'a3' : {'boundaries' : [30000, 60000], 'values' : [1e-3, 3e-4, 1e-4]},
                'a4' : {'boundaries' : [30000, 60000], 'values' : [3e-3, 1e-3, 3e-4]},
                'a5' : {'boundaries' : [30000, 60000], 'values' : [1e-2, 3e-3, 1e-3]},
                'b1' : {'boundaries' : [80000, 140000], 'values' : [1e-4, 3e-5, 1e-5]},
                'b2' : {'boundaries' : [80000, 140000], 'values' : [3e-4, 1e-4, 3e-5]},
                'b3' : {'boundaries' : [80000, 140000], 'values' : [1e-3, 3e-4, 1e-4]},
                'b4' : {'boundaries' : [80000, 140000], 'values' : [3e-3, 1e-3, 3e-4]},
                'b5' : {'boundaries' : [80000, 140000], 'values' : [1e-2, 3e-3, 1e-3]}
                 },
            128 : {
                'a1' : {'boundaries' : [60000, 120000], 'values' : [1e-4, 3e-5, 1e-5]},
                'a2' : {'boundaries' : [60000, 120000], 'values' : [3e-4, 1e-4, 3e-5]},
                'a3' : {'boundaries' : [60000, 120000], 'values' : [1e-3, 3e-4, 1e-4]},
                'a4' : {'boundaries' : [60000, 120000], 'values' : [3e-3, 1e-3, 3e-4]},
                'a5' : {'boundaries' : [60000, 120000], 'values' : [1e-2, 3e-3, 1e-3]}
                 },
            64 : {
                'a1' : {'boundaries' : [120000, 240000], 'values' : [1e-4, 3e-5, 1e-5]},
                'a2' : {'boundaries' : [120000, 240000], 'values' : [3e-4, 1e-4, 3e-5]},
                'a3' : {'boundaries' : [120000, 240000], 'values' : [1e-3, 3e-4, 1e-4]},
                'a4' : {'boundaries' : [120000, 240000], 'values' : [3e-3, 1e-3, 3e-4]},
                'a5' : {'boundaries' : [120000, 240000], 'values' : [1e-2, 3e-3, 1e-3]}
                 },
            32 : {
                'a1' : {'boundaries' : [240000, 480000], 'values' : [1e-4, 3e-5, 1e-5]},
                'a2' : {'boundaries' : [240000, 480000], 'values' : [3e-4, 1e-4, 3e-5]},
                'a3' : {'boundaries' : [240000, 480000], 'values' : [1e-3, 3e-4, 1e-4]},
                'a4' : {'boundaries' : [240000, 480000], 'values' : [3e-3, 1e-3, 3e-4]},
                'a5' : {'boundaries' : [240000, 480000], 'values' : [1e-2, 3e-3, 1e-3]}
                 }
        }
    }
COLORS = 'bgrcmykw'
MARKERS = 'o1s*+xD^v'

ACTIVATE_K_SET = np.arange(1, 5)
K_SET = [1,4,16]
#===========================Local configuration==============================#
RESULT_DIR = '../../../exp_results/imagnet_exps/hash_metric2/'
RESULT_DIR_READ = '../../../exp_results/imagnet_exps/hash_metric1/'
#========================PARAM============================#
DATASET= 'imgnet32'
GPU_ID = 0
BATCH_SIZE = 256
NSCLASS = 16
EPOCH = 100
# model
EMBED_M= 64
CONV_NAME = 'conv2'
LOSS_TYPE = 'npair'
# hash
BUCKET_D = 64
ACTIVATE_K = 1
# hash loss
HASH_LOSS_TYPE = 'npair'
HASH_DISTANCE_TYPE = 'hash1'
HASH_LAMBDA = 0.003
HASH_MARGIN_ALPHA = 1.0
# hash learning
HASH_DECAY_TYPE = 'stair'
HASH_DECAY_PARAM_TYPE = 'type1' 
# minimum cost flow
PAIRWISE_LAMBDA = 0.01 

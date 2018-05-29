import sys
sys.path.append('../../configs')
sys.path.append('../../utils')
sys.path.append('../../tfops')

# ../../utils
from datasetmanager import DATASETMANAGER_DICT
from format_op import params2id, listformat1, listformat2, numformat2
from csv_op import CsvWriter2, CsvWriter, read_csv
from writer import create_muldir, write_pkl
from reader import read_pkl

# ./
from local_config import ACTIVATE_K_SET, K_SET,\
                         DATASET, GPU_ID, BATCH_SIZE, NSCLASS, EPOCH,\
                         CONV_NAME, EMBED_M, LOSS_TYPE,\
                         BUCKET_D, ACTIVATE_K,\
                         HASH_DECAY_TYPE, HASH_DECAY_PARAM_TYPE,\
                         HASH_LOSS_TYPE, HASH_DISTANCE_TYPE,\
                         HASH_LAMBDA, HASH_MARGIN_ALPHA,\
                         PAIRWISE_LAMBDA,\
                         COLORS, RESULT_DIR, RESULT_DIR_READ
from deepmetric import DeepMetric

import numpy as np
import itertools
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", default = GPU_ID, help="Utilize which gpu", type = int)
parser.add_argument("--dataset", default = DATASET, help="dataset to be used", type = str)
parser.add_argument("--nbatch", default = BATCH_SIZE, help="size of batch used to train", type = int)
parser.add_argument("--nsclass", default = NSCLASS, help="the number of selected class", type = int)
parser.add_argument("--epoch", default = EPOCH, help="epoch to be ran", type = int)
parser.add_argument("--conv", default = CONV_NAME, help="conv network", type = str)
parser.add_argument("--ltype", default = LOSS_TYPE, help="loss type", type = str)
parser.add_argument("--m", default = EMBED_M, help="embedding m", type = int)
parser.add_argument("--hltype", default = HASH_LOSS_TYPE, help="hash loss type", type = str)
parser.add_argument("--hdtype", default = HASH_DECAY_TYPE, help="decay type", type = str)
parser.add_argument("--hdptype", default = HASH_DECAY_PARAM_TYPE, help="hash decay param type", type = str)
parser.add_argument("--hdt", default = HASH_DISTANCE_TYPE, help="hash distance type", type = str)
parser.add_argument("--hma", default = HASH_MARGIN_ALPHA, help="hash margin alpha", type = float)
parser.add_argument("--hlamb", default = HASH_LAMBDA, help="hash lambda", type = float)
parser.add_argument("--plamb", default = PAIRWISE_LAMBDA, help="lambda for pairwise cost", type = float)
parser.add_argument("--k", default = ACTIVATE_K, help="activate k", type = int)
parser.add_argument("--d", default = BUCKET_D, help="bucket d", type = int)

args = parser.parse_args()

if __name__ == '__main__':
    args.ltype=args.hltype
    args.m = args.d
    if args.ltype=='npair':
        args.nsclass = args.nbatch//2 

    FILE_ID = params2id(args.dataset, args.conv, args.ltype, args.m)

    if args.hltype=='npair':
        args.nsclass = args.nbatch//2
        assert args.nsclass==args.nbatch//2, "In npair loss, nsclass should be half of batch size"

    if args.hltype=='npair':
        HASH_FILE_ID = FILE_ID+'_'+params2id(args.nbatch, args.nsclass, args.hltype, args.hdt, args.d, args.k, args.hlamb, args.plamb, args.hdtype, args.hdptype)
    else:
        HASH_FILE_ID = FILE_ID+'_'+params2id(args.nbatch, args.nsclass, args.hltype, args.hdt, args.d, args.k, args.hma, args.plamb, args.hdtype, args.hdptype)

    SAVE_DIR = RESULT_DIR_READ+'metric/save/%s/'%FILE_ID
    HASH_LOG_DIR = './log/'
    HASH_CSV_DIR = RESULT_DIR+'hash/csv/'
    HASH_PKL_DIR = RESULT_DIR+'hash/pkl/'
    HASH_ASSET_DIR = RESULT_DIR+'hash/asset/'
    HASH_SAVE_DIR = RESULT_DIR+'hash/save/%s/'%HASH_FILE_ID
    HASH_BOARD_DIR = RESULT_DIR+'hash/board/%s/'%HASH_FILE_ID

    create_muldir(HASH_LOG_DIR)
    datasetmanager = DATASETMANAGER_DICT[args.dataset]
    # load data
    dm_train, dm_val, dm_test = datasetmanager(args.hltype, nsclass=args.nsclass) if args.hltype in ['triplet', 'npair'] else datasetmanager(args.hltype)
    dm_train.print_shape()
    dm_val.print_shape()
    dm_test.print_shape()

    model = DeepMetric(dm_train, dm_val, dm_test, HASH_LOG_DIR+HASH_FILE_ID+'.log', args)
    model.build(False)
    model.restore(save_dir=SAVE_DIR)
    model.prepare_test()
    model.build_hash()
    model.set_up_train_hash()
    model.restore_hash(save_dir=HASH_SAVE_DIR)
    model.prepare_test_hash()
    performance_hash = model.test_hash_metric(args.k, K_SET)
    model.delete()
    del model
    del dm_train
    del dm_val
    del dm_test

    write_pkl(performance_hash, path=HASH_PKL_DIR + HASH_FILE_ID+'.pkl')
    cwrite = CsvWriter2(1) 
    key_set = [
        'test_nmi',
        'te_te_suf',
        'te_te_precision_at_k',
        'te_te_recall_at_k',
        'te_te_precision_ir', 
        'te_te_recall_ir'
        ]
    for key in key_set:
        cwrite.add_header(0, str(key))
        content = ''
        if 'suf' in str(key):
            content = listformat1(performance_hash[key])
        elif 'at_k' in str(key):
            content = listformat2(performance_hash[key])
        elif 'ir' in str(key):
            content = numformat2(performance_hash[key], "%.3f")
        else:
            content = performance_hash[key]
        cwrite.add_content(0, content)
    cwrite.write(HASH_CSV_DIR+HASH_FILE_ID+'.csv')


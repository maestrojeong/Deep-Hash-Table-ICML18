import sys
sys.path.append('../../configs')
sys.path.append('../../utils')
sys.path.append('../../tfops')

# ../../utils
from datasetmanager import DATASETMANAGER_DICT
from format_op import params2id, listformat
from csv_op import CsvWriter2, CsvWriter
from writer import create_muldir, write_pkl

# ./
from local_config import K_SET, RESULT_DIR,\
                         DATASET, GPU_ID, BATCH_SIZE, EPOCH, NSCLASS,\
                         CONV_NAME, LOSS_TYPE, EMBED_M, BUCKET_D,\
                         HASH_LOSS_TYPE, HASH_DISTANCE_TYPE,\
                         PAIRWISE_LAMBDA, HASH_LAMBDA, HASH_MARGIN_ALPHA,\
                         HASH_DECAY_TYPE, HASH_DECAY_PARAM_TYPE

from deepmetric import DeepMetric

import numpy as np
import itertools
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", default = GPU_ID, help="Utilize which gpu", type = int)
parser.add_argument("--dataset", default = DATASET, help="dataset to be used", type = str)
parser.add_argument("--nbatch", default = BATCH_SIZE, help="size of batch", type = int)
parser.add_argument("--nsclass", default = NSCLASS, help="the number of selected class", type = int)
parser.add_argument("--epoch", default = EPOCH, help="epoch to be ran", type = int)
parser.add_argument("--conv", default = CONV_NAME, help="conv network", type = str)
parser.add_argument("--ltype", default = LOSS_TYPE, help="loss type", type = str)
parser.add_argument("--m", default = EMBED_M, help="embedding m", type = int)
parser.add_argument("--d", default = BUCKET_D, help="bucket d", type = int)
parser.add_argument("--hltype", default = HASH_LOSS_TYPE, help="hash loss type", type = str)
parser.add_argument("--hdt", default = HASH_DISTANCE_TYPE, help="hash distance type", type = str)
parser.add_argument("--plamb", default = PAIRWISE_LAMBDA, help="lambda for pairwise cost", type = float)
parser.add_argument("--hlamb", default = HASH_LAMBDA, help="hash lambda", type = float)
parser.add_argument("--hma", default = HASH_MARGIN_ALPHA, help="hash margin alpha", type = float)
parser.add_argument("--hdtype", default = HASH_DECAY_TYPE, help="decay type", type = str)
parser.add_argument("--hdptype", default = HASH_DECAY_PARAM_TYPE, help="hash decay param type", type = str)

args = parser.parse_args()

if __name__ == '__main__':
    args.ltype=args.hltype
    args.m = args.d

    FILE_ID = params2id(args.dataset, args.conv, args.ltype, args.m)

    if args.hltype=='npair': args.nsclass = args.nbatch//2

    HASH_METRIC_PARAM = args.hlamb if args.hltype=='npair' else args.hma
    HASH_FILE_ID = params2id(FILE_ID, args.nbatch, args.nsclass, args.hltype, args.hdt, args.d, args.k, HASH_METRIC_PARAM, args.plamb, args.hdtype, args.hdptype)

    SAVE_DIR = RESULT_DIR+'metric/save/%s/'%FILE_ID
    HASH_SAVE_DIR = RESULT_DIR+'exp1/save/%s/'%HASH_FILE_ID
    LOG_DIR = RESULT_DIR+'exp1/log/'
    CSV_DIR = RESULT_DIR+'exp1/csv/'
    PKL_DIR = RESULT_DIR+'exp1/pkl/'
    BOARD_DIR = RESULT_DIR+'exp1/board/%s/'%HASH_FILE_ID

    create_muldir(SAVE_DIR, LOG_DIR, CSV_DIR, PKL_DIR, HASH_SAVE_DIR, BOARD_DIR)

    # load data
    datasetmanager = DATASETMANAGER_DICT[args.dataset]
    dm_train, dm_val, dm_test = datasetmanager(args.hltype, nsclass=args.nsclass)
    for v in [dm_train, dm_val, dm_test]: v.print_shape()

    model = DeepMetric(dm_train, dm_val, dm_test, LOG_DIR+HASH_FILE_ID+'.log', args)
    model.build()
    model.restore(save_dir=SAVE_DIR)
    model.prepare_test()
    model.build_hash()
    model.set_up_train_hash()

    try: 
        model.restore_hash(save_dir=HASH_SAVE_DIR)
    except AttributeError:
        model.initialize()
        model.train_hash(200, save_dir=HASH_SAVE_DIR, board_dir=BOARD_DIR)
        model.restore_hash(save_dir=HASH_SAVE_DIR)

    model.prepare_test_hash()
    performance_hash = model.test_hash_metric(args.k, K_SET)

    model.delete()
    del model
    del dm_train
    del dm_val
    del dm_test

    write_pkl(performance_hash, path=PKL_DIR + HASH_FILE_ID+'_train.pkl')
    cwrite = CsvWriter2(1) 
    key_set = ['train_nmi', 'te_tr_suf', 'te_tr_precision_at_k', 'te_tr_recall_at_k']
    for key in key_set:
        cwrite.add_header(0, str(key))
        content = ''
        if 'suf' in str(key): content = listformat(performance_hash[key])
        elif 'at_k' in str(key): content = listformat(performance_hash[key])
        else: content = performance_hash[key]
        cwrite.add_content(0, content)
    cwrite.write(CSV_DIR+HASH_FILE_ID+'_train.csv')

    write_pkl(performance_hash, path=PKL_DIR + HASH_FILE_ID+'_test.pkl')
    cwrite = CsvWriter2(1) 
    key_set = ['test_nmi', 'te_te_suf', 'te_te_precision_at_k', 'te_te_recall_at_k']
    for key in key_set:
        cwrite.add_header(0, str(key))
        content = ''
        if 'suf' in str(key): content = listformat(performance_hash[key])
        elif 'at_k' in str(key): content = listformat(performance_hash[key])
        else: content = performance_hash[key]
        cwrite.add_content(0, content)
    cwrite.write(CSV_DIR+HASH_FILE_ID+'_test.csv')

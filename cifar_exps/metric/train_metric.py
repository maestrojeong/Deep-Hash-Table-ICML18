import sys
sys.path.append('../../configs')
sys.path.append('../../utils')
sys.path.append('../../tfops')

# ../../utils
from datasetmanager import DATASETMANAGER_DICT
from format_op import params2id, listformat
from csv_op import CsvWriter2, read_csv
from writer import create_muldir, write_pkl

# ./
from local_config import K_SET, RESULT_DIR, \
                         DATASET, GPU_ID, BATCH_SIZE, EPOCH, NSCLASS,\
                         CONV_NAME, EMBED_M,\
                         LOSS_TYPE, MARGIN_ALPHA, LAMBDA,\
                         DECAY_TYPE, DECAY_PARAM_TYPE
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
parser.add_argument("--lamb", default = LAMBDA, help="lambda", type = float)
parser.add_argument("--ma", default = MARGIN_ALPHA, help="margin alpha", type = float)
parser.add_argument("--dtype", default = DECAY_TYPE, help="decay type", type = str)
parser.add_argument("--dptype", default = DECAY_PARAM_TYPE, help="decay parameter type", type = str)

args = parser.parse_args() # parameter required for model

nk = len(K_SET)

if __name__ == '__main__':
    if args.ltype=='npair': args.nsclass = args.nbatch//2 

    #for args.lamb in [0.001, 0.003, 0.01]
    #for args.ma in [0.3, 1.0]:

    METRIC_PARAM = args.lamb if args.ltype=='npair' else args.ma
    FILE_ID = params2id(args.dataset, args.nbatch, args.nsclass, args.conv, args.ltype, METRIC_PARAM, args.m, args.dtype, args.dptype)

    SAVE_DIR = RESULT_DIR+'metric/save/%s/'%FILE_ID
    LOG_DIR = RESULT_DIR+'metric/log/'
    CSV_DIR = RESULT_DIR+'metric/csv/'
    PKL_DIR = RESULT_DIR+'metric/pkl/'
    BOARD_DIR = RESULT_DIR+'metric/board/%s/'%FILE_ID
    create_muldir(SAVE_DIR, BOARD_DIR, LOG_DIR, CSV_DIR, PKL_DIR)

    # load data
    datasetmanager= DATASETMANAGER_DICT[args.dataset]
    dm_train, dm_val, dm_test = datasetmanager(args.ltype, nsclass=args.nsclass)
    for v in [dm_train, dm_val, dm_test]: v.print_shape()

    model = DeepMetric(dm_train, dm_val, dm_test, LOG_DIR+FILE_ID+'.log', args)
    model.build()
    model.set_up_train()
    try:
        model.restore(save_dir=SAVE_DIR)
    except AttributeError:
        model.initialize()
        model.train(epoch=args.epoch, save_dir=SAVE_DIR, board_dir=BOARD_DIR)
        model.restore(save_dir=SAVE_DIR)
    model.prepare_test()

    performance = model.test_metric(K_SET)
    key_set = ['te_tr_recall_at_k', 'te_tr_precision_at_k', 'te_te_recall_at_k', 'te_te_precision_at_k']
    cwrite = CsvWriter2(1) 
    for key in key_set:
        cwrite.add_header(0, str(key))
        cwrite.add_content(0, listformat(performance[key]))
    cwrite.write(CSV_DIR+FILE_ID+'.csv')
    write_pkl(performance, path=PKL_DIR + FILE_ID+'.pkl')


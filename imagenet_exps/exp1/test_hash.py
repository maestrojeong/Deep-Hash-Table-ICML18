import sys
sys.path.append('../../configs')
sys.path.append('../../utils')
sys.path.append('../../tfops')

# ../../utils
from datasetmanager import DATASETMANAGER_DICT
from format_op import params2id, listformat
from shutil_op import remove_file, remove_dir, copy_file, copy_dir
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
import shutil
import glob
import os
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

nactivate = len(ACTIVATE_K_SET)

if __name__ == '__main__':
    args.m=args.d
    args.ltype = args.hltype

    FILE_ID = params2id(args.dataset, args.conv, args.ltype, args.m)

    if args.hltype=='npair': args.nsclass = args.nbatch//2 

    HASH_FILE_ID = params2id(FILE_ID, args.hltype, args.hdt, args.d, args.k)
    QUERY_FILE_ID = params2id(FILE_ID, '*', '*', args.hltype, args.hdt, args.d, args.k, '*', '*', '*', '*')

    print("file id : {}".format(HASH_FILE_ID))
    print("query id : {}".format(QUERY_FILE_ID))

    PKL_DIR = RESULT_DIR+'exp1/pkl/'
    CSV_DIR = RESULT_DIR+'exp1/csv/'
    SAVE_DIR = RESULT_DIR+'exp1/save/%s/'%HASH_FILE_ID

    copy_dst_csv = CSV_DIR+HASH_FILE_ID+'.csv'
    copy_dst_pkl = PKL_DIR+HASH_FILE_ID+'.pkl'

    if os.path.exists(SAVE_DIR): remove_dir(SAVE_DIR)
    if os.path.exists(copy_dst_csv): remove_file(copy_dst_csv)
    if os.path.exists(copy_dst_pkl): remove_file(copy_dst_pkl)

    pkl_files = glob.glob(PKL_DIR+QUERY_FILE_ID+'.pkl')
    print(pkl_files)
    if len(pkl_files)==0:
        print("No such pkl files")
        sys.exit() 

    best_file_id = os.path.basename(pkl_files[0])[:-4] # -.pkl'
    best_performance = np.sum(read_pkl(pkl_files[0])['te_te_precision_at_k'])
    for pkl_idx in range(len(pkl_files)):
        file_id = os.path.basename(pkl_files[pkl_idx])[:-4] # -.pkl'
        performance = np.sum(read_pkl(pkl_files[pkl_idx])['te_te_precision_at_k'])
        print("performance : {} from {}".format(performance, file_id))
        if performance > best_performance:
            best_performance, best_file_id = performance, file_id
    print("best performance : {} from {}".format(best_performance, best_file_id))

    copy_file(CSV_DIR+best_file_id+'.csv', copy_dst_csv)
    copy_file(PKL_DIR+best_file_id+'.pkl', copy_dst_pkl)
    copy_dir(RESULT_DIR+'exp1/save/'+best_file_id, SAVE_DIR)


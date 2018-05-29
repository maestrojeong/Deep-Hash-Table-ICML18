import sys
sys.path.append('../../configs')
sys.path.append('../../utils')
sys.path.append('../../tfops')

# ../../configs
from info import ACTIVATE_K_SET

# ../../utils
from datasetmanager import DATASETMANAGER_DICT
from shutil_op import remove_file, remove_dir, copy_file, copy_dir
from format_op import params2id, listformat
from writer import create_muldir, write_pkl
from csv_op import CsvWriter2
from reader import read_pkl

# ./
from local_config import K_SET, RESULT_DIR,\
                         DATASET, GPU_ID, BATCH_SIZE, EPOCH, NSCLASS,\
                         CONV_NAME, EMBED_M,\
                         LOSS_TYPE, MARGIN_ALPHA, LAMBDA,\
                         DECAY_TYPE, DECAY_PARAM_TYPE

from deepmetric import DeepMetric
import numpy as np
import itertools

import glob
import os
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

args = parser.parse_args()

if __name__ == '__main__':
    args.d=args.m
    FILE_ID = params2id(args.dataset, args.conv, args.ltype, args.m)
    QUERY_FILE_ID = params2id(args.dataset, '*', '*', args.conv, args.ltype, '*', args.m, '*', '*')

    CSV_DIR = RESULT_DIR+'metric/csv/'
    PKL_DIR = RESULT_DIR+'metric/pkl/'
    LOG_DIR = RESULT_DIR+'metric/log/'
    SAVE_DIR = RESULT_DIR+'metric/save/%s/'%FILE_ID

    create_muldir(PKL_DIR, CSV_DIR, LOG_DIR)

    copy_dst_csv = CSV_DIR+FILE_ID+'.csv'
    copy_dst_pkl = PKL_DIR+FILE_ID+'.pkl'

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
            best_performance = performance
            best_file_id = file_id
    print("best performance : {} from {}".format(best_performance, best_file_id))
    copy_file(CSV_DIR+best_file_id+'.csv', copy_dst_csv)
    copy_file(PKL_DIR+best_file_id+'.pkl', copy_dst_pkl)
    copy_dir(RESULT_DIR+'metric/save/'+best_file_id, SAVE_DIR)

    # load data
    datasetmanager= DATASETMANAGER_DICT[args.dataset]
    dm_train, dm_val, dm_test = datasetmanager(args.ltype, nsclass=args.nsclass)
    for v in [dm_train, dm_val, dm_test]: v.print_shape()

    model = DeepMetric(dm_train, dm_val, dm_test, LOG_DIR+FILE_ID+'.log', args)
    model.build()
    model.set_up_train()
    model.restore(save_dir=SAVE_DIR)
    model.prepare_test()

    for activate_k in ACTIVATE_K_SET:
        performance_th = model.test_th(activate_k, K_SET)
        performance_vq = model.test_vq(activate_k, K_SET)

        write_pkl({'th' : performance_th, 'vq' : performance_vq}, path=PKL_DIR + FILE_ID+'_{}.pkl'.format(activate_k))

        cwrite = CsvWriter2(2) 

        key_set = [
            'train_nmi',
            'test_nmi',
            'te_tr_suf',
            'te_te_suf',
            'te_tr_precision_at_k',
            'te_te_precision_at_k',
            'te_tr_recall_at_k',
            'te_te_recall_at_k'
            ]

        for key in key_set:
            cwrite.add_header(0, str(key)+"_th")
            cwrite.add_header(1, str(key)+"_vq")

            content = ''
            if 'suf' in str(key): content = listformat(performance_th[key])
            elif 'at_k' in str(key): content = listformat(performance_th[key])
            else: content = performance_th[key]
            cwrite.add_content(0, content)

            content = ''
            if 'suf' in str(key): content = listformat(performance_vq[key])
            elif 'at_k' in str(key): content = listformat(performance_vq[key])
            else: content = performance_vq[key]
            cwrite.add_content(1, content)
        cwrite.write(CSV_DIR+FILE_ID+'_{}.csv'.format(activate_k))

    model.delete()

import sys
sys.path.append('../../configs')
sys.path.append('../../utils')
sys.path.append('../../tfops')

# ../../utils
from format_op import params2id, listformat1, listformat2, numformat1, numformat2
from csv_op import CsvWriter, read_csv
from writer import create_muldir, write_pkl
from reader import read_pkl

# ./
from local_config import ACTIVATE_K_SET, K_SET,\
                         DATASET, GPU_ID,\
                         CONV_NAME, PRETRAIN, EMBED_M,\
                         LOSS_TYPE, MARGIN_ALPHA, LAMBDA,\
                         DECAY_TYPE, DECAY_PARAM_TYPE, HASH_DECAY_TYPE, HASH_DECAY_PARAM_TYPE,\
                         BUCKET_D, ACTIVATE_K, FINE_TUNE,\
                         HASH_METRIC_SHAPE, HASH_DISTANCE_TYPE, HASH_MARGIN_ALPHA,\
                         HASH_LAMBDA, HASH_MARGIN_ALPHA,\
                         COLORS, MARKERS, RESULT_DIR, RESULT_DIR_READ

from deepmetric import DeepMetric

from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", default = GPU_ID, help="Utilize which gpu", type = int)
parser.add_argument("--dataset", default = DATASET, help="dataset to be used", type = str)
parser.add_argument("--conv", default = CONV_NAME, help="conv network", type = str)
parser.add_argument("--ltype", default = LOSS_TYPE, help="loss type", type = str)
parser.add_argument("--m", default = EMBED_M, help="embedding m", type = int)
parser.add_argument("--lamb", default = LAMBDA, help="lambda", type = float)
parser.add_argument("--ma", default = MARGIN_ALPHA, help="margin alpha", type = float)
parser.add_argument("--dtype", default = DECAY_TYPE, help="decay type", type = str)
parser.add_argument("--dptype", default = DECAY_PARAM_TYPE, help="decay parameter type", type = str)
parser.add_argument("--hdtype", default = HASH_DECAY_TYPE, help="decay type", type = str)
parser.add_argument("--hdptype", default = HASH_DECAY_PARAM_TYPE, help="hash decay param type", type = str)
parser.add_argument("--hdt", default = HASH_DISTANCE_TYPE, help="hash distance type", type = str)
parser.add_argument("--hms", default = HASH_METRIC_SHAPE, help="hash metric shape", type = str)
parser.add_argument("--hma", default = HASH_MARGIN_ALPHA, help="hash margin alpha", type = float)
parser.add_argument("--hlamb", default = HASH_LAMBDA, help="hash lambda", type = float)
parser.add_argument("--k", default = ACTIVATE_K, help="activate k", type = int)
parser.add_argument("--d", default = BUCKET_D, help="bucket d", type = int)

args = parser.parse_args()

nactivate = len(ACTIVATE_K_SET)
nk = len(K_SET)

if __name__ == '__main__':
    args.m=args.d
    if args.ltype=='npair':
        FILE_ID = params2id(args.dataset, args.conv, args.ltype, args.lamb, args.m, PRETRAIN)
        EVAL_FILE_ID = FILE_ID+'_'+params2id(args.hdt, args.hms, args.d, args.hlamb, FINE_TUNE)
    else:
        FILE_ID = params2id(args.dataset, args.conv, args.ltype, args.ma, args.m, PRETRAIN)
        EVAL_FILE_ID = FILE_ID+'_'+params2id(args.hdt, args.hms, args.d, args.hma, FINE_TUNE)

    PKL_DIR = RESULT_DIR_READ+'metric/pkl/'
    HASH_ASSET_DIR = RESULT_DIR+'hash/asset/'
    HASH_CSV_DIR = RESULT_DIR+'hash/csv/'
    HASH_PKL_DIR = RESULT_DIR+'hash/pkl/'
    create_muldir(HASH_ASSET_DIR, HASH_CSV_DIR, HASH_PKL_DIR)

    performance_hash_total = dict()
    performance_th_total = dict()
    performance_vq_total = dict()

    performance_metric = read_pkl(PKL_DIR+FILE_ID+'.pkl')

    key_set = [
        'train_nmi',
        'test_nmi',
        'te_tr_suf',
        'te_te_suf',
        'te_tr_precision_at_k',
        'te_te_precision_at_k',
        'te_tr_recall_at_k',
        'te_te_recall_at_k',
        'te_tr_precision_ir',
        'te_te_precision_ir', 
        'te_tr_recall_ir',
        'te_te_recall_ir']

    for key in key_set:
        performance_hash_total[key] = list()
        performance_th_total[key] = list()
        performance_vq_total[key] = list()

    for activate_k_index in range(len(ACTIVATE_K_SET)):
        activate_k = ACTIVATE_K_SET[activate_k_index]
        if args.ltype=='npair':
            HASH_FILE_ID = FILE_ID+'_'+params2id(args.hdt, args.hms, args.d, activate_k, args.hlamb, FINE_TUNE)
        else:
            HASH_FILE_ID = FILE_ID+'_'+params2id(args.hdt, args.hms, args.d, activate_k, args.hma, FINE_TUNE)

        performance_hash = read_pkl(HASH_PKL_DIR+HASH_FILE_ID+'.pkl')

        for key in key_set:
            performance_hash_total[key].append(performance_hash[key])

        content = read_pkl(PKL_DIR+FILE_ID+'_{}.pkl'.format(activate_k))

        performance_th = content['th']
        performance_vq = content['vq']
        for key in key_set:
            performance_th_total[key].append(performance_th[key])

        for key in key_set:
            performance_vq_total[key].append(performance_vq[key])
                         
    write_pkl({
        'metric' : performance_metric,
        'th' : performance_th,
        'vq' : performance_vq,
        'hash' : performance_hash
        }, path=HASH_PKL_DIR+EVAL_FILE_ID+'.pkl')

    pdf = PdfPages(HASH_ASSET_DIR+EVAL_FILE_ID+'.pdf')
    plt.figure(figsize=[17,17])
    # plot NMI

    for key in key_set: 
        performance_th_total[key] = np.array(performance_th_total[key])
        performance_vq_total[key] = np.array(performance_vq_total[key])
        performance_hash_total[key] = np.array(performance_hash_total[key])

    for key in key_set:
        if 'nmi' in str(key):
            plt.title(str(key))
            plt.plot(ACTIVATE_K_SET, performance_th_total[key], color=COLORS[1%len(COLORS)], linestyle='-', marker=MARKERS[1%len(MARKERS)], label="th")
            for x, y in zip(ACTIVATE_K_SET, performance_th_total[key]):
                plt.text(x, y, numformat1(y, "%.3f"))
            plt.plot(ACTIVATE_K_SET, performance_vq_total[key], color=COLORS[2%len(COLORS)], linestyle='-', marker=MARKERS[2%len(MARKERS)], label="vq")
            for x, y in zip(ACTIVATE_K_SET, performance_vq_total[key]):
                plt.text(x, y, numformat1(y, "%.3f"))
            plt.plot(ACTIVATE_K_SET, performance_hash_total[key], color=COLORS[3%len(COLORS)], linestyle='-', marker=MARKERS[3%len(MARKERS)], label="hash")
            for x, y in zip(ACTIVATE_K_SET, performance_hash_total[key]):
                plt.text(x, y, numformat1(y, "%.3f"))
            plt.legend()
            pdf.savefig()

    for key in key_set:
        if 'ir' in str(key):
            plt.title(str(key))
            plt.plot(ACTIVATE_K_SET, performance_th_total[key], color=COLORS[1%len(COLORS)], linestyle='-', marker=MARKERS[1%len(MARKERS)], label="th")
            for x, y in zip(ACTIVATE_K_SET, performance_th_total[key]):
                plt.text(x, y, numformat1(y, "%.3f"))
            plt.plot(ACTIVATE_K_SET, performance_vq_total[key], color=COLORS[2%len(COLORS)], linestyle='-', marker=MARKERS[2%len(MARKERS)], label="vq")
            for x, y in zip(ACTIVATE_K_SET, performance_vq_total[key]):
                plt.text(x, y, numformat1(y, "%.3f"))
            plt.plot(ACTIVATE_K_SET, performance_hash_total[key], color=COLORS[3%len(COLORS)], linestyle='-', marker=MARKERS[3%len(MARKERS)], label="hash")
            for x, y in zip(ACTIVATE_K_SET, performance_hash_total[key]):
                plt.text(x, y, numformat1(y, "%.3f"))
            plt.legend()
            pdf.savefig()

    for key in key_set:
        if 'at_k' in str(key):
            for k_idx in range(nk):
                plt.title("{}@{}".format(str(key),str(K_SET[k_idx])))
                plt.plot(ACTIVATE_K_SET, performance_metric[k_idx]*np.ones(nactivate), color=COLORS[0%len(COLORS)], linestyle='--', marker=MARKERS[0%len(MARKERS)], label="baseline")
                for x, y in zip(ACTIVATE_K_SET, performance_metric[k_idx]*np.ones(nactivate)):
                    plt.text(x, y, numformat1(y, "%.3f"))
                plt.plot(ACTIVATE_K_SET, performance_th_total[key][k_idx], color=COLORS[1%len(COLORS)], linestyle='-', marker=MARKERS[1%len(MARKERS)], label="th")
                for x, y in zip(ACTIVATE_K_SET, performance_th_total[key]):
                    plt.text(x, y, numformat1(y, "%.3f"))
                plt.plot(ACTIVATE_K_SET, performance_vq_total[key][k_idx], color=COLORS[2%len(COLORS)], linestyle='-', marker=MARKERS[2%len(MARKERS)], label="vq")
                for x, y in zip(ACTIVATE_K_SET, performance_vq_total[key]):
                    plt.text(x, y, numformat1(y, "%.3f"))
                plt.plot(ACTIVATE_K_SET, performance_hash_total[key][k_idx], color=COLORS[3%len(COLORS)], linestyle='-', marker=MARKERS[3%len(MARKERS)], label="hash")
                for x, y in zip(ACTIVATE_K_SET, performance_hash_total[key]):
                    plt.text(x, y, numformat1(y, "%.3f"))
                plt.legend()
                pdf.savefig()
           
    for key in key_set:
        if 'suf' in str(key):
            for k_idx in range(nk):
                plt.title("{}@{}".format(str(key),str(K_SET[k_idx])))
                plt.plot(ACTIVATE_K_SET, performance_th_total[key][k_idx], color=COLORS[1%len(COLORS)], linestyle='-', marker=MARKERS[1%len(MARKERS)], label="th")
                for x, y in zip(ACTIVATE_K_SET, performance_th_total[key]):
                    plt.text(x, y, numformat1(y, "%.3f"))
                plt.plot(ACTIVATE_K_SET, performance_vq_total[key][k_idx], color=COLORS[2%len(COLORS)], linestyle='-', marker=MARKERS[2%len(MARKERS)], label="vq")
                for x, y in zip(ACTIVATE_K_SET, performance_vq_total[key]):
                    plt.text(x, y, numformat1(y, "%.3f"))
                plt.plot(ACTIVATE_K_SET, performance_hash_total[key][k_idx], color=COLORS[3%len(COLORS)], linestyle='-', marker=MARKERS[3%len(MARKERS)], label="hash")
                for x, y in zip(ACTIVATE_K_SET, performance_hash_total[key]):
                    plt.text(x, y, numformat1(y, "%.3f"))
                plt.legend()
                pdf.savefig()

    def get_F1score(recall_array, precision_array):
        assert len(recall_array)==len(precision_array), "Wrong argument"
        length = len(recall_array)
        f1array = np.zeros(length)

        for i in range(length):
            f1array[i] = 2*recall_array[i]*precision_array[i]/(recall_array[i]+precision_array[i])
        return f1array

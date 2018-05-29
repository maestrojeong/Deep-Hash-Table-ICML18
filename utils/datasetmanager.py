import sys
sys.path.append('../utils')
sys.path.append('../configs')

# ../configs
from path import CIFARPROCESSED,IMAGENET32PROCESSED
from info import CIFARNCLASS, IMAGENETCLASS

# ../utils
from reader import read_npy, read_pkl
from datamanager import DATAMANAGER_DICT

import numpy as np

def cifar_manager(dm_type='basic', nsclass=0):
    '''
    Args:
        dm_type - string 
            defaults to be basic
        nsclass - int
            which required when dm_type='triplet'
    Return:
        dm_train, dm_val, dm_test
            datamanager for each set
    '''
    assert dm_type in DATAMANAGER_DICT.keys(), "The type of data should be in {}".format(DATAMANAGER_DICT.keys())

    dm = DATAMANAGER_DICT[dm_type]

    train_input = read_npy(CIFARPROCESSED+'train_image.npy')
    train_label = read_npy(CIFARPROCESSED+'train_label.npy')
    val_input = read_npy(CIFARPROCESSED+'val_image.npy')
    val_label = read_npy(CIFARPROCESSED+'val_label.npy')
    test_input = read_npy(CIFARPROCESSED+'test_image.npy')
    test_label = read_npy(CIFARPROCESSED+'test_label.npy')

    if dm_type in ['triplet', 'npair']: 
        dm_train = dm(train_input, train_label, CIFARNCLASS, nsclass) 
        dm_val = dm(val_input, val_label, CIFARNCLASS, nsclass)
        dm_test = dm(test_input, test_label, CIFARNCLASS, nsclass) 
    else:
        dm_train = dm(train_input, train_label, CIFARNCLASS)    
        dm_val = dm(val_input, val_label, CIFARNCLASS)    
        dm_test = dm(test_input, test_label, CIFARNCLASS)    

    return dm_train, dm_val, dm_test

def imagenet32_manager(dm_type='basic', nsclass=0):
    '''
    Args:   
        dm_type - string
        nsclass -int
    '''
    train_img = read_npy(IMAGENET32PROCESSED+'train_img.npy')
    train_label = read_npy(IMAGENET32PROCESSED+'train_label.npy')
    val_img = read_npy(IMAGENET32PROCESSED+'val_img.npy')
    val_label = read_npy(IMAGENET32PROCESSED+'val_label.npy')
    test_img = read_npy(IMAGENET32PROCESSED+'test_img.npy')
    test_label = read_npy(IMAGENET32PROCESSED+'test_label.npy')

    dm = DATAMANAGER_DICT[dm_type]
    if dm_type in ['triplet', 'npair']: 
        dm_train = dm(train_img, train_label, IMAGENETCLASS, nsclass) 
        dm_val = dm(val_img, val_label, IMAGENETCLASS, nsclass)
        dm_test = dm(test_img, test_label, IMAGENETCLASS, nsclass) 
    else:
        dm_train = dm(train_img, train_label, IMAGENETCLASS)    
        dm_val = dm(val_img, val_label, IMAGENETCLASS)    
        dm_test = dm(test_img, test_label, IMAGENETCLASS)    

    return dm_train, dm_val, dm_test

DATASETMANAGER_DICT = {
        'cifar' : cifar_manager,\
        'imgnet32' : imagenet32_manager
        }

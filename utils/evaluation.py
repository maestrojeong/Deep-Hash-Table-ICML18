from eval_op import HashTable, get_recall_precision_at_k
from format_op import listformat

import numpy as np

def evaluate_metric_te_tr(test_label, train_label, te_te_distance, te_tr_distance, k_set, logger=None):
    '''
    Args:
        test_label - Numpy 1D array [ntest]
        train_label - Numpy 1D array [ntrain]
        te_te_distance - Numpy 2D array [ntest, ntest]
        te_tr_distance - Numpy 2D array [ntest, ntrain]
        k_set - list
        logger - logger
            defaults to be None
    Return:
        performance - dict
    '''
    te_te_recall_at_k, te_te_precision_at_k = get_recall_precision_at_k(dist_matrix=te_te_distance, labelq=test_label, labelh=test_label, k_set=k_set, issame=True)
    te_tr_recall_at_k, te_tr_precision_at_k = get_recall_precision_at_k(dist_matrix=te_tr_distance, labelq=test_label, labelh=train_label, k_set=k_set, issame=False)

    performance = {
                'te_tr_precision_at_k' : te_tr_precision_at_k,
                'te_te_precision_at_k' : te_te_precision_at_k,
                'te_tr_recall_at_k' : te_tr_recall_at_k,
                'te_te_recall_at_k' : te_te_recall_at_k}
    key_set = ['te_tr_recall_at_k', 'te_tr_precision_at_k', 'te_te_recall_at_k', 'te_te_precision_at_k']
           
    for key in key_set:
        content = '{} @ {} =  {}'.format(str(key), listformat(k_set), listformat(performance[key]))
        if logger is None:
            print(content)
        else:
            logger.info(content)

    return performance

def evaluate_metric_te(test_label, te_te_distance, k_set, logger=None):
    '''
    Args:
        test_label - Numpy 1D array [ntest]
        te_te_distance - Numpy 2D array [ntest, ntest]
        k_set - list
        logger - logger
            defaults to be None
    Return:
        performance - dict
    '''
    te_te_recall_at_k, te_te_precision_at_k = get_recall_precision_at_k(dist_matrix=te_te_distance, labelq=test_label, labelh=test_label, k_set=k_set, issame=True)

    performance = {
                'te_te_precision_at_k' : te_te_precision_at_k,
                'te_te_recall_at_k' : te_te_recall_at_k
                }
    key_set = ['te_te_recall_at_k', 'te_te_precision_at_k']
           
    for key in key_set:
        content = '{} @ {} =  {}'.format(str(key), listformat(k_set), listformat(performance[key]))
        if logger is None:
            print(content)
        else:
            logger.info(content)

    return performance

def evaluate_hash_te(test_hash_key, te_te_distance, te_te_query_key, te_te_query_value, test_label, ncls_test, activate_k, k_set, logger=None):
    '''
    Args:
        test_hash_key - Numpy 2D array [ntest, nbucket]
            should be binary
        te_te_distance - Numpy 2D array [ntest, ntest]
        te_te_query_key - Numpy 2D array [ntest, nbucket] 
            should be binary
        te_te_query_value - Numpy 2D array [ntest, nbucket] 
            continuous
        test_label - label of test data [ntest] 
        ncls_test - int
            number of test classes or labels
        activate_k - int
            num of activation buckets
        k_set - list [nk]
        logger - logger
            defaults to be None
    Return: 
        performance - dict
    '''
    test_hash_table = HashTable(hash_key=test_hash_key, labelh=test_label, nlabel=ncls_test) 
    test_nmi = test_hash_table.get_nmi()

    te_te_srr, te_te_recall_at_k, te_te_precision_at_k = test_hash_table.get_srr_recall_precision_at_k_hash(dist_matrix=te_te_distance, query_key=te_te_query_value, labelq=test_label, base_activate_k=activate_k, k_set=k_set, issame=True)
    te_te_suf = 1.0/np.mean(te_te_srr, axis=-1)

    performance = {
            'test_nmi' : test_nmi,
            'te_te_suf' : te_te_suf,
            'te_te_precision_at_k' : te_te_precision_at_k,
            'te_te_recall_at_k' : te_te_recall_at_k
            }

    key_set = [
        'test_nmi',
        'te_te_suf',
        'te_te_precision_at_k',
        'te_te_recall_at_k'
        ]

    for key in key_set:
        if 'suf' in str(key): content = "{} {} = {}".format(str(key), listformat(k_set), listformat(performance[key])) 
        elif 'at_k' in str(key): content = "{} {} = {}".format(str(key), listformat(k_set), listformat(performance[key])) 
        else: content = "{} = {:.3f}".format(str(key), performance[key])
        
        if logger is None: print(content)
        else: logger.info(content)

    return performance

def evaluate_hash_te_tr(train_hash_key, test_hash_key, te_tr_distance, te_te_distance, te_tr_query_key, te_tr_query_value, te_te_query_key, te_te_query_value, train_label, test_label, ncls_train, ncls_test, activate_k, k_set, logger=None):
    '''
    Args:
        train_hash_key - Numpy 2D array [ntrain, nbucket]
            should be binary
        test_hash_key - Numpy 2D array [ntest, nbucket]
            should be binary
        te_tr_distance - Numpy 2D array [ntest, ntrain]
        te_te_distance - Numpy 2D array [ntest, ntest]
        te_tr_query_key - Numpy 2D array [ntest, nbucket] 
            should be binary
        te_tr_query_value - Numpy 2D array [ntest, nbucket] 
            continuous
        te_te_query_key - Numpy 2D array [ntest, nbucket] 
            should be binary
        te_te_query_value - Numpy 2D array [ntest, nbucket] 
            continuous
        train_label - label of train data [ntrain]
        test_label - label of test data [ntest] 
        ncls_train - int
            number of train classes or labels
        ncls_test - int
            number of test classes or labels
        activate_k - int
            num of activation buckets
        k_set - list [nk]
        logger - logger
            defaults to be None

    Return: 
        performance - dict
    '''
    train_hash_table = HashTable(hash_key=train_hash_key, labelh=train_label, nlabel=ncls_train) 
    test_hash_table = HashTable(hash_key=test_hash_key, labelh=test_label, nlabel=ncls_test) 

    train_nmi = train_hash_table.get_nmi()
    test_nmi = test_hash_table.get_nmi()

    te_te_srr, te_te_recall_at_k, te_te_precision_at_k = test_hash_table.get_srr_recall_precision_at_k_hash(dist_matrix=te_te_distance, query_key=te_te_query_value, labelq=test_label, base_activate_k=activate_k, k_set=k_set, issame=True)
    te_tr_srr, te_tr_recall_at_k, te_tr_precision_at_k = train_hash_table.get_srr_recall_precision_at_k_hash(dist_matrix=te_tr_distance, query_key=te_tr_query_value, labelq=test_label, base_activate_k=activate_k, k_set=k_set, issame=False)
    
    te_tr_suf = 1.0/np.mean(te_tr_srr, axis=-1)
    te_te_suf = 1.0/np.mean(te_te_srr, axis=-1)

    performance = {
            'train_nmi' : train_nmi,
            'test_nmi' : test_nmi,
            'te_tr_suf' : te_tr_suf,
            'te_te_suf' : te_te_suf,
            'te_tr_precision_at_k' : te_tr_precision_at_k,
            'te_te_precision_at_k' : te_te_precision_at_k,
            'te_tr_recall_at_k' : te_tr_recall_at_k,
            'te_te_recall_at_k' : te_te_recall_at_k
            }

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
        if 'suf' in str(key): content = "{} {} = {}".format(str(key), listformat(k_set), listformat(performance[key])) 
        elif 'at_k' in str(key): content = "{} {} = {}".format(str(key), listformat(k_set), listformat(performance[key])) 
        else: content = "{} = {:.3f}".format(str(key), performance[key])
        
        if logger is None: print(content)
        else: logger.info(content)

    return performance


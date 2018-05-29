import numpy as np

def shuffle_img_label(img, label):
    '''
    Dependency : numpy as np
    Args:
        img - numpy ND array
        label - numpy 1D array
    Return:
        shuffled img, shuffled label
            just shuffled same shape
    '''
    ndata = len(label)
    idx = np.arange(ndata)
    np.random.shuffle(idx)
    return img[idx], label[idx]

def print_dict_shape(dict_):
    for key in dict_.keys():
        if isinstance(dict_[key], np.ndarray):
            print("{} : {}".format(key, dict_[key].shape))


def label_count(labels, nclass):
    '''
    Dependency : numpy as np
    Args:
        labels - 1D numpy int array
        nclass - int
            label \in [0, nclass)
    Return:
        label_counter    
    '''

    label_counter = np.zeros(nclass)
    for index in range(len(labels)): label_counter[labels[index]]+=1
    return label_counter

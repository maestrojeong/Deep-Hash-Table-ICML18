from tqdm import tqdm

import tensorflow as tf
import numpy as np

def np_random_crop_4d(imgs, size):
    '''
    Args:
        imgs - 4d image NHWC
        size - list (rh, rw)
    '''
    rh, rw = size
    
    on, oh, ow, oc = imgs.shape

    cropped_imgs = np.zeros([on, rh, rw, oc])

    ch = np.random.randint(low=0, high=oh-rh, size=on)
    cw = np.random.randint(low=0, high=ow-rw, size=on)

    for idx in range(on):
        cropped_imgs[idx] = imgs[idx,ch[idx]:ch[idx]+rh,cw[idx]:cw[idx]+rw]
    
    return cropped_imgs

def activate_k_1D(arr, k): 
    ''' activate top k-bit to 1 otherwise 0
    Dependency: numpy as np
    Args:
        arr - numpy 1D array
        k - int
    Return:
        arr_k_active - numpy 1D array
    '''
    length = len(arr)
    arr_k_active = np.zeros_like(arr)

    index = np.argsort(-arr)
    index = index[:k] # select highest k 

    arr_k_active[index] = 1
    return arr_k_active

def activate_k_2D(arr, k, session=None, batch_size=100):
    '''activate top k-bit to 1 otherwise 0
    Dependency: activate_k_1D, numpy as np
    Args:
        arr - numpy 2D array
        k - int
        session - session of tensorflow
            defaults to be None
        batch_size - int
            defautls to be 100
    Return 
        arr_k_active - numpy 2D array
    '''
    if session is None:
        arr_k_active = np.zeros_like(arr)
        ndata = len(arr)

        for i in range(ndata):
            arr_k_active[i] = activate_k_1D(arr=arr[i], k=k)

        return arr_k_active
    else:
        ndata, nfeature = arr.shape
        inputs = tf.placeholder(tf.float32, [batch_size, nfeature]) # [batch_size, nfeature]
        top_k = tf.nn.top_k(inputs, k=k+1)[0] # [batch_size, k+1]
        kth_element = 0.5*tf.add(top_k[:,k-1], top_k[:,k]) # [batch_size]
        kth_element = tf.reshape(kth_element, [-1,1]) # [batch_size, 1]
        k_hash_tensor = tf.cast(0.5*(tf.sign(tf.subtract(inputs, kth_element))+1), tf.int32) # [batch_size, nfeatures]
        
        if ndata%batch_size!=0:
            arr = np.concatenate([arr, np.zeros([batch_size-ndata%batch_size, nfeature])], axis=0) 
        nbatch=len(arr)//batch_size
        arr_k_active = list()
        for b in tqdm(range(nbatch), ascii=True, desc="batch"):
            feed_dict = {inputs : arr[b*batch_size:(b+1)*batch_size]}
            arr_k_active.append(session.run(k_hash_tensor, feed_dict=feed_dict))
        arr_k_active = np.concatenate(arr_k_active, axis=0)
        arr_k_active = arr_k_active[:ndata]
        return arr_k_active

def hamming_dist(arr1, arr2):
    '''hamming distance between two 1D arrays(arr1 and arr2)
    Dependency: numpy as np
    Args:
        arr1 - 1D numpy array with only 0 and 1
        arr2 - 1D numpy array with only 0 and 1
    Return:
        h_d - float hamming distance between arr1 and arr2
    '''
    assert len(arr1)==len(arr2), "Length of arr1 and arr2 should be same but %d and %d"%(len(arr1), len(arr2))

    length = len(arr1)
    h_d = 0
    for i in range(length):
        if arr1[i]!=arr2[i]:
            h_d+=1
    return h_d

def pairwise_distance_euclid_np(components, identity=0):
    '''Get the inter distances between components with metric
    Dependency: numpy as np
    Args:
        components - Numpy 2D array [ndata, nfeature]
        identity - float 
    Return:
        dist_matrix - 2D numpy array [n, n]
                        dist_matrix[i,j] = metric(components[i], components[j])
                        dist_matrix[i,i] = identity
    '''
    ndata, nfeature = components.shape
    dist_matrix = np.zeros([ndata, ndata])

    for i in tqdm(range(ndata), ascii=True, desc="idx"):
        for j in range(i, ndata):
            dist_matrix[i][j] = np.sum(np.square(components[i]-components[j]))
            dist_matrix[j][i] = dist_matrix[i][j]

    for i in range(ndata):
        dist_matrix[i][i] = identity
    return dist_matrix

def pairwise_dist(components, metric, identity=0):
    '''Get the inter distances between components with metric
    Dependency: numpy as np
    Args:
        components - list or numpy n sets
        metric - func(x1, x2) for the distance between x1 and x2
                 metric(a,b) = metric(b,a)
        identity - float 
    Return:
        dist_matrix - 2D numpy array [n, n]
                        dist_matrix[i,j] = metric(components[i], components[j])
                        dist_matrix[i,i] = identity
    '''
    ndata = len(components)
    dist_matrix = np.zeros([ndata, ndata])
    for i in range(ndata):
        for j in range(i,ndata):
            dist_matrix[i][j] = metric(components[i], components[j])
            dist_matrix[j][i] = dist_matrix[i][j]

    for i in range(ndata):
        dist_matrix[i][i] = identity
    return dist_matrix

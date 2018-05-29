from tensorflow.python.framework import dtypes, ops, sparse_tensor, tensor_shape
from tensorflow.python.ops import array_ops, control_flow_ops, logging_ops, math_ops,\
                                  nn, script_ops, sparse_ops

import tensorflow as tf
import numpy as np

from tqdm import tqdm

def apply_tf_op(inputs, session, input_gate, output_gate, batch_size, dim=4, train_gate=None):
    '''
    Requires the graph to be built alreadly
    Dependency:
        import tensorflow as tf
        import numpy as np
        from tqdm import tqdm

    Args:
        inputs - 2-D vector [ndata, nfeature]
                 4-D image [ndata, height, width, nchannel]
        session - session of tf to run
        input_gate - placeholder for tf operation
        output_gate - output tensor
        batch_size - int
        dim - int
        train_gate - determine whether train or not
    Return:
        outputs - N-D image [ndata]
    '''
    if dim == 4:
        ndata, height, width, nchannel = inputs.shape
        if ndata%batch_size!=0:
            inputs = np.concatenate([inputs, np.zeros([batch_size-ndata%batch_size, height, width, nchannel])], axis=0) 
    else:
        ndata, nfeature = inputs.shape
        if ndata%batch_size!=0:
            inputs = np.concatenate([inputs, np.zeros([batch_size-ndata%batch_size, nfeature])], axis=0) 

    nbatch = len(inputs)//batch_size

    outputs = list()
    for b in tqdm(range(nbatch), ascii = True, desc="batch"):
        feed_dict = {input_gate : inputs[b*batch_size:(b+1)*batch_size]}
        if train_gate is not None:
            feed_dict[train_gate] = False 
        outputs.append(session.run(output_gate, feed_dict=feed_dict))
    outputs = np.concatenate(outputs, axis=0)
    outputs = outputs[:ndata]
    return outputs

def pairwise_distance_euclid_efficient(input1, input2, session, batch_size): 
    '''
    Args:
        input1 - Numpy 2D array [ndata1, nfeature]
        input2 - Numpy 2D array [ndata2, nfeature]
    '''
    assert input1.shape[1]==input2.shape[1], "input1, input2 should have same feature"

    ndata1, nfeature = input1.shape
    ndata2, _ = input2.shape

    input1_s = tf.placeholder(tf.float32, shape=[batch_size, nfeature])
    
    input2_t = tf.convert_to_tensor(input2, dtype=tf.float32)
    
    p_dist = math_ops.add(
    	math_ops.reduce_sum(math_ops.square(input1_s), axis=[1], keep_dims=True),
        math_ops.reduce_sum(
                math_ops.square(array_ops.transpose(input2_t)), axis=[0], keep_dims=True))-\
                2.0 * math_ops.matmul(input1_s, array_ops.transpose(input2_t))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    p_dist = math_ops.maximum(p_dist, 0.0)
    error_mask = math_ops.less_equal(p_dist, 0.0)
    p_dist = math_ops.multiply(p_dist, math_ops.to_float(math_ops.logical_not(error_mask)))

    return apply_tf_op(inputs=input1, session=session, input_gate=input1_s, output_gate=p_dist, batch_size=batch_size, dim=2)

def get_recall_at_1_efficient(data, label, input1_tensor, input2_tensor, idx_tensor, session): 
    '''
    Args:
        data - Numpy 2D array [ndata, nfeature]
        label - Numpy 1D array [ndata]
        input1_tensor - placeholder [batch_size, nfeature]
        input2_tensor - placeholder [ndata, nfeature]
        idx_tensor - tensor [ndata, 2]
    '''
    batch_size = input1_tensor.get_shape().as_list()[0]
    ndata, nfeature = data.shape

    if ndata%batch_size!=0:
        inputs = np.concatenate([data, np.zeros([batch_size-ndata%batch_size, nfeature])], axis=0) 
    nbatch = len(inputs)//batch_size

    outputs = np.zeros([len(inputs), 2])
    for b in tqdm(range(nbatch), ascii = True, desc="batch"):
        feed_dict = {
                    input1_tensor : inputs[b*batch_size:(b+1)*batch_size],\
                    input2_tensor : data
                    }
        outputs[b*batch_size:(b+1)*batch_size]=session.run(idx_tensor, feed_dict=feed_dict)
    outputs = outputs[:ndata] # [ndata, 2]

    nsuccess = 0
    for idx1 in range(ndata):
        for idx2 in outputs[idx1]:
            if int(idx1)==int(idx2):
                continue
            if label[int(idx1)]==label[int(idx2)]: 
                nsuccess+=1
                break
    return nsuccess/ndata

if __name__=='__main__':
    from dist import pairwise_distance_euclid_v2
    
    sess= tf.Session()
    array1 = np.random.uniform(size=[5, 10])
    array2 = np.random.uniform(size=[5, 10])
    print(pairwise_distance_euclid_efficient(array1, array2, session=sess, batch_size=100))
    print(sess.run(pairwise_distance_euclid_v2(tf.convert_to_tensor(array1, dtype=tf.float32), tf.convert_to_tensor(array2, dtype=tf.float32))))
    





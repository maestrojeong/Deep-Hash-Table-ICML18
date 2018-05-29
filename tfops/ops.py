import tensorflow as tf
import numpy as np

def clip_ts(tensor, vmax=1.0, vmin=-1.0):
    return tf.clip_by_value(tensor, clip_value_min=vmin, clip_value_max=vmax)

def get_shape(x):
    '''get the shape of tensor as list'''
    return x.get_shape().as_list()

def vars_info_vl(var_list): 
    return "    "+"\n    ".join(["{} : {}".format(v.name, get_shape(v)) for v in var_list])

def vars_info(string):
    '''print variables in collection named string'''
    return "Collection name %s\n"%string+vars_info_vl(tf.get_collection(string))

def get_initialized_vars(sess):
    var_list = list()
    for var in tf.global_variables():
        try: sess.run(var)
        except tf.errors.FailedPreconditionError: continue
        var_list.append(var)
    return var_list

def rest_initializer(sess):
    print("Initialize uninitialized variables")
    uninitailized_variables=[]
    for var in tf.global_variables():
        try :
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitailized_variables.append(var)
    sess.run(tf.variables_initializer(uninitailized_variables))
    print("Variables initialized")
    print(vars_info_vl(uninitailized_variables))

def full_initializer(sess):
    print("Initialize all variables")
    sess.run(tf.global_initializer())

def get_train_op(optimizer, loss, lr, var_list=tf.trainable_variables()):
    '''
    Args:
        optimizer - tf optimizer
                ex) tf.train.AdamOptimizer(1e-4) 
        loss - a tensor
        lr - float
            learning rate
        var_list - list of tensors
    Return:
        train_op
    '''
    return optimizer(lr).minimize(loss=loss, var_list=var_list)

def get_multi_train_op(optimizer, loss, lr_list, vl_list):
    '''
    Args:
        optimizer - tf optimizer
                ex) tf.train.AdamOptimizer 
        loss - a tensor
        lr_list - learning rate list
        vl_list - list of variable list
    Return:
        train_op
    '''
    assert len(lr_list)==len(vl_list), "The length of lr_list, and vl_list should be same but %d and %d"%(len(lr_list), len(vl_list))
    nlist = len(lr_list)
    opt_list = list()
    grad_list = list()
    train_op_list = list()
   

    def list_summer(list_):
        v = list_[0]
        for i in range(1, len(list_)):
            v=v+list_[i]
        return v
    grads = tf.gradients(loss, list_summer(vl_list))

    for i in range(nlist):
        opt_list.append(optimizer(lr_list[i]))

    start = 0
    for i in range(nlist):
        grad_list.append(grads[start:start+len(vl_list[i])]) 
        train_op_list.append(opt_list[i].apply_gradients(zip(grad_list[i], vl_list[i])))
        start+=len(vl_list[i])
    train_op = tf.group(*train_op_list)

    return train_op 

def count_vars(var_list):
    '''count the # of vars in var_list'''
    count = 0
    for var in var_list:
        var_shape = get_shape(var)
        var_size = 1
        for size in var_shape:
            var_size*=size
        count+=var_size
    return count

def sigmoid_cross_entropy(labels, logits):
    '''
    Args:
        labels - N-D tensor
        logits - N-D tensor
    '''
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

def softmax_cross_entropy(labels, logits):
    '''
    Args:
        labels - (N-1)-D tensor int32 or int64
        logits - N-D tensor float32 
    '''
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def pool2d(x, per_op, op_name):
    assert op_name in ['AVG', 'MAX'], "op_name arg should be 'AVG' or 'MAX'"

    if op_name == 'AVG':
        return tf.nn.pool(x, window_shape = per_op, pooling_type = 'AVG', strides = per_op, padding='VALID')
    else:
        return tf.nn.pool(x, window_shape = per_op, pooling_type = 'MAX', strides = per_op, padding='VALID')

def max_pool2d(x, per_max):
    '''
    max_pool2d, wrapper of pool2d
    Args:
        x - 4d tensor
            'NHWC' batch, height, width, channel
        per_max - list with two ints
    '''
    return pool2d(x, per_op=per_max, op_name='MAX')

def avg_pool2d(x, per_avg):
    '''
    avg_pool2d wrapper of pool2d
    Args:
        x - 4d tensor
            'NHWC' batch, height, width, channel
        per_avg - list with two ints
    '''
    return pool2d(x, per_op=per_avg, op_name='AVG')

def conv2d(input_, filter_shape, strides = [1,1,1,1], padding = False, activation = None, batch_norm = False, istrain = False, scope = None):
    '''
    Args:
        input_ - 4D tensor
            Normally NHWC format
        filter_shape - 1D array 4 elements
            [height, width, inchannel, outchannel]
        strides - 1D array 4 elements
            default to be [1,1,1,1]
        padding - bool
            Deteremines whether add padding or not
            True => add padding 'SAME'
            Fale => no padding  'VALID'
        activation - activation function
            default to be None
        batch_norm - bool
            default to be False
            used to add batch-normalization
        istrain - bool
            indicate the model whether train or not
        scope - string
            default to be None
    Return:
        4D tensor
        activation(batch(conv(input_)))
    '''
    with tf.variable_scope(scope or "conv"):
        if padding:
            padding = 'SAME'
        else:
            padding = 'VALID'
        w = tf.get_variable(name="weight", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)) 
        conv = tf.nn.conv2d(input_, w, strides=strides, padding=padding)
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay = 0.99, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="bias", shape = filter_shape[-1], initializer=tf.constant_initializer(0.001))
            if activation is None:
                return conv + b
            return activation(conv + b)

def fc_layer(input_, output_size, activation = None, batch_norm = False, istrain = False, scope = None):
    '''
    fully convlolution layer
    Args :
        input_  - 2D tensor
            general shape : [batch, input_size]
        output_size - int
            shape of output 2D tensor
        activation - activation function
            defaults to be None
        batch_norm - bool
            defaults to be False
            if batch_norm to apply batch_normalization
        istrain - bool
            defaults to be False
            indicator for phase train or not
        scope - string
            defaults to be None then scope becomes "fc"
    '''
    with tf.variable_scope(scope or "fc"):
        w = tf.get_variable(name="weight", shape = [get_shape(input_)[1], output_size], initializer=tf.contrib.layers.xavier_initializer())
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(tf.matmul(input_, w) , center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="bias", shape = [output_size], initializer=tf.constant_initializer(0.0))
            if activation is None:
                return tf.nn.xw_plus_b(input_, w, b)
            return activation(tf.nn.xw_plus_b(input_, w, b))

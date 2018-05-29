import sys
sys.path.append('../../pretrain')

# pretrain

from ops import max_pool2d,\
                conv2d, fc_layer,\
                get_shape
                
from augment_op import random_distort, random_crop, random_flip_rl, resize

import tensorflow as tf

slim = tf.contrib.slim

def conv1(x, is_training, reuse=False):
    '''
    Args:
        x - 4D tensor [batch, 32, 32, 3] 
        is_training - bool tensor
        reuse - bool

    Return:
        last, logits 
            last - 2D tensor
            logits - 2D tensor
    '''
    r = tf.cond(
            is_training, 
            lambda : random_flip_rl(
                        random_crop(
                            tf.pad(
                                x, [[0,0],[4,4],[4,4],[0,0]], 
                                "CONSTANT"), 32, 32)),
            lambda : x)

    def leaky_relu_custom(alpha):
        def func(features):
            return tf.nn.leaky_relu(features, 1/alpha)
        return func

    with tf.variable_scope("NIN", reuse=reuse):
        with slim.arg_scope([slim.conv2d], weights_initializer=tf.contrib.slim.variance_scaling_initializer(), weights_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=leaky_relu_custom(5.5)):
                with slim.arg_scope(([slim.dropout]), is_training=is_training, keep_prob=0.5):
                    n = r
                    n = slim.conv2d(n, 192, [5, 5], scope='conv2d_0')
                    n = slim.conv2d(n, 160, [1, 1], scope='conv2d_1')
                    n = slim.conv2d(n, 96, [1, 1], scope='conv2d_2')
                    n = slim.max_pool2d(n, [3,3], stride=2, padding='SAME')
                    n = slim.dropout(n)
                    n = slim.conv2d(n, 192, [5, 5], scope='conv2d_3')
                    n = slim.conv2d(n, 192, [1, 1], scope='conv2d_4')
                    n = slim.conv2d(n, 192, [1, 1], scope='conv2d_5')
                    n = slim.avg_pool2d(n, [3,3], stride=2, padding='SAME')
                    n = slim.dropout(n)
                    n = slim.conv2d(n, 192, [3, 3], scope='conv2d_6')
                    n = slim.conv2d(n, 192, [1, 1], activation_fn=None, scope='conv2d_7')
                    last = tf.reduce_mean(n, [1,2])
                    n = tf.nn.relu(n)
                    n = slim.conv2d(n, 100, [1,1], activation_fn=None, scope='conv2d_8')
                    logits = tf.reduce_mean(n, [1,2])
                    return last, logits
#=============================================================================================================================================#
CONV_DICT = {
            'cifar': 
                {
                'conv1' : conv1
                },
            'imgnet32': 
                {
                'conv1' : conv1
                },
            'icml_cifar': 
                {
                'conv1' : conv1
                }
            }
#=============================================================MANAGER====================================================================#

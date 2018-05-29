from ops import get_shape

import tensorflow as tf


def rotate_90_cc(inputs):
    '''
    Args:
        inputs - 4D tensor [batch, height, width, nchannel]
    Return:
        rotates - 4D tensor [batch, width, height, nchannel]
    rotates[i][width-k][j][l] = inputs[i][j][k][l]
    '''
    rotates = tf.transpose(inputs, [0,2,1,3]) 
    rotates = tf.reverse(rotates, axis=[1])    
    return rotates
    
def flip_rl(inputs):
    '''
    Args:
        inputs - 4D tensor [batch, height, width, nchannel]
    Return:
        flips - 4D tensor [batch, height, width, nchannel]

    flips[i][j][width-k][l] = inputs[i][j][k][l]
    '''
    flips = tf.reverse(inputs, axis=[2])
    return flips

def flip_td(inputs):
    '''
    flip top down
    Args:
        inputs - 4D tensor [batch, height, width, nchannel]
    Return:
        flips - 4D tensor [batch, height, width, nchannel]

    flips[i][j][width-k][l] = inputs[i][j][k][l]
    '''
    flips = tf.reverse(inputs, axis=[1])
    return flips


def resize(inputs, nh, nw, method='bilinear'):
    '''
        resize iamges
    '''
    return tf.image.resize_images(inputs, [nh, nw])

def random_rotate_90_cc(inputs):
    '''
    For square size image random rotate_90_cc
    Args: 
        inputs - 4D tensor
    Return: 
        rotate 90 counter clockwise or original image
    '''
    rand = tf.random_uniform([], minval=-1, maxval=1)
    return tf.cond(tf.greater(rand, 0.0), lambda: inputs, lambda: rotate_90_cc(inputs))

def random_flip_rl(inputs):
    '''
    Args: 
        inputs - 4D tensor
    Return: 
        flip left right or original image
    '''
    rand = tf.random_uniform([], minval=-1, maxval=1)
    return tf.cond(tf.greater(rand, 0.0), lambda: inputs, lambda: flip_rl(inputs))

def random_flip_td(inputs):
    '''
    Args: 
        inputs - 4D tensor
    Return: 
        flip top down or original image
    '''
    rand = tf.random_uniform([], minval=-1, maxval=1)
    return tf.cond(tf.greater(rand, 0.0), lambda: inputs, lambda: flip_td(inputs))

def random_distort(inputs):
    '''
    The image should be square
    Args:
        inputs - 4D tensor
    Return :
        8 possibility randomly goes out
    '''
    return random_rotate_90_cc(random_flip_rl(random_flip_td(inputs)))

def random_crop(inputs, ch, cw):
    '''
    apply tf.random_crop on 4D
    Args:
        inputs- 4D tensor
        ch - int
            crop height
        cw - int
            crop width
    '''
    ib, ih, iw, ic = get_shape(inputs)
    return tf.random_crop(inputs, [ib, ch, cw, ic])

if __name__ == '__main__':
    def get_shape(tensor):
        return tensor.get_shape().as_list()

    a = tf.constant([[1,2],[3,4]], dtype=tf.float32)
    a = tf.reshape(a, [1,2,2,1])
    a = tf.tile(a, [2,1,1,2])

    b = tf.random_uniform([1,2,3,4])
    sess = tf.Session()
    print("Value")
    print(sess.run(a[0,:,:,0]))
    print("rotate_90_cc")
    print(sess.run(rotate_90_cc(a)[0,:,:,0]))
    print("flip_rl")
    print(sess.run(flip_rl(a)[0,:,:,0]))

    print("Shape")
    print(get_shape(b))
    print("rotate_90_cc")
    print(get_shape(rotate_90_cc(b)))
    print("flip_rl")
    print(get_shape(flip_rl(b)))

    a_random = random_distort(a)
    print("random")
    for i in range(15):
        print(sess.run(a_random[0,:,:,0]))

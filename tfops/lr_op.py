import tensorflow as tf

def stair_decay(initial_lr, decay_steps, decay_rate, initial_step=0):
    '''
    Args:
        initial_lr - float
        decay_steps - int
        decay_rate - float
        initial_step - int
    Return : 
        learing_rate - self decaying
            initial_lr*decay_rate^int(global_step/decay_steps)
        update_step_op - tf op
            add 1 global step
    '''
    global_step = tf.Variable(initial_step, trainable=False)
    update_step_op = tf.assign_add(global_step, 1)
    return tf.train.exponential_decay(
                learning_rate=initial_lr,\
                global_step=global_step,\
                decay_steps=decay_steps,\
                decay_rate=decay_rate,\
                staircase=True), update_step_op

def piecewise_decay(boundaries, values, initial_step = 0):
    '''
    Args:
        initial_step - int defaults to be 0
        boundaries - list with int 
        values - list with float
    Return : 
        learing_rate - self decaying
            
        update_step_op - tf op
            add 1 global step
    '''
    global_step = tf.Variable(initial_step, name='global_step', trainable=False)
    update_step_op = tf.assign_add(global_step, 1)
    return tf.train.piecewise_constant(global_step, boundaries, values), update_step_op

DECAY_DICT = {
            'stair' : stair_decay,
            'piecewise' : piecewise_decay
            }


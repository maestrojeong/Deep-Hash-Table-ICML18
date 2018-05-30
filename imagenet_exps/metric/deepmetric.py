import sys
sys.path.append('../../configs')
sys.path.append('../../utils')
sys.path.append('../../tfops')

# ../../utils 
from evaluation import evaluate_metric_te, evaluate_hash_te
from sklearn_op import KMeansClustering
from logger_op import LoggerManager
from gpu_op import selectGpuById
from np_op import activate_k_2D

# ../../tfops
from transform_op import apply_tf_op, pairwise_distance_euclid_efficient, get_recall_at_1_efficient
from summary_op import SummaryWriter
from lr_op import DECAY_DICT
from nets import CONV_DICT
from ops import rest_initializer, vars_info_vl, get_multi_train_op, get_initialized_vars
from dist import npairs_loss, triplet_semihard_loss,\
                 pairwise_distance_euclid, pairwise_distance_euclid_v2


# ./
from local_config import DECAY_PARAMS_DICT
from tqdm import tqdm

import tensorflow as tf
slim = tf.contrib.slim

from tensorflow.python.ops import array_ops, math_ops

import numpy as np
import glob
import os

class DeepMetric:
    def __init__(self, train_dataset, val_dataset, test_dataset, logfilepath, args):
        self.args = args

        selectGpuById(self.args.gpu)
        self.logfilepath = logfilepath
        self.logger = LoggerManager(self.logfilepath, __name__)

        self.dataset_dict = dict()
        self.set_train_dataset(train_dataset) 
        self.set_val_dataset(val_dataset) 
        self.set_test_dataset(test_dataset) 

    def set_train_dataset(self, train_dataset):
        self.logger.info("Setting train_dataset starts")
        self.train_dataset = train_dataset
        self.dataset_dict['train'] = self.train_dataset
        self.ntrain, self.height, self.width, self.nchannel = self.train_dataset.image.shape
        self.ncls_train = self.train_dataset.nclass
        self.nbatch_train = self.ntrain//self.args.nbatch
        self.logger.info("Setting train_dataset ends")

    def set_test_dataset(self, test_dataset):
        self.logger.info("Setting test_dataset starts")
        self.test_dataset = test_dataset
        self.dataset_dict['test'] = self.test_dataset
        self.test_image = self.dataset_dict['test'].image 
        self.test_label = self.dataset_dict['test'].label
        self.ntest = self.test_dataset.ndata
        self.ncls_test = self.test_dataset.nclass
        self.nbatch_test = self.ntest//self.args.nbatch
        self.logger.info("Setting test_dataset ends")

    def set_val_dataset(self, val_dataset):
        self.logger.info("Setting val_dataset starts")
        self.val_dataset = val_dataset
        self.dataset_dict['val'] = self.val_dataset
        self.val_image = self.dataset_dict['val'].image 
        self.val_label = self.dataset_dict['val'].label
        self.nval = self.val_dataset.ndata
        self.nbatch_val = self.nval//self.args.nbatch
        self.logger.info("Setting val_dataset ends")
    
    def switch_log_path(self, logfilepath):
        self.logger.remove()
        print("Log file switched from {} to {}".format(self.logfilepath, logfilepath))
        self.logfilepath = logfilepath
        self.logger = LoggerManager(self.logfilepath, __name__)

    def build(self):
        self.logger.info("Model building starts")
        tf.reset_default_graph()
        if self.args.ltype == 'npair':
            self.anc_img = tf.placeholder(tf.float32, shape = [self.args.nbatch//2, self.height, self.width, self.nchannel])
            self.pos_img = tf.placeholder(tf.float32, shape = [self.args.nbatch//2, self.height, self.width, self.nchannel])
            self.istrain = tf.placeholder(tf.bool, shape= [])
            self.label = tf.placeholder(tf.int32, shape = [self.args.nbatch//2])
        else: # triplet
            self.img = tf.placeholder(tf.float32, shape = [self.args.nbatch, self.height, self.width, self.nchannel])
            self.istrain = tf.placeholder(tf.bool, shape= [])
            self.label = tf.placeholder(tf.int32, shape = [self.args.nbatch])
        
        self.generate_sess()

        self.conv_net = CONV_DICT[self.args.dataset][self.args.conv]

        if self.args.ltype == 'npair':
            self.anc_last, _ = self.conv_net(self.anc_img, is_training=self.istrain, reuse=False)
            self.pos_last, _ = self.conv_net(self.pos_img, is_training=self.istrain, reuse=True)
            self.anc_last = tf.nn.relu(self.anc_last)
            self.pos_last = tf.nn.relu(self.pos_last)
        else:#triplet
            self.last, _ = self.conv_net(self.img, is_training=self.istrain, reuse=False)
            self.last = tf.nn.relu(self.last)

        with slim.arg_scope([slim.fully_connected], activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0005), biases_initializer=tf.zeros_initializer()):
            if self.args.ltype == 'npair':
                with tf.variable_scope('Embed', reuse=False): self.anc_embed = slim.fully_connected(self.anc_last, self.args.m, scope="fc1")
                with tf.variable_scope('Embed', reuse=True): self.pos_embed = slim.fully_connected(self.pos_last, self.args.m, scope="fc1")
                self.loss = npairs_loss(labels=self.label, embeddings_anchor=self.anc_embed, embeddings_positive=self.pos_embed, reg_lambda=self.args.lamb)
            else:#triplet
                with tf.variable_scope('Embed', reuse=False): self.embed = slim.fully_connected(self.last, self.args.m, scope = "fc1")
                self.embed_l2_norm = tf.nn.l2_normalize(self.embed, dim=-1) # embedding with l2 normalization
                def pairwise_distance_c(embeddings): return pairwise_distance_euclid(embeddings, squared=True)
                self.loss = triplet_semihard_loss(labels=self.label, embeddings=self.embed_l2_norm, pairwise_distance=pairwise_distance_c, margin=self.args.ma)

        self.loss += tf.losses.get_regularization_loss()
        initialized_variables=get_initialized_vars(self.sess)
        self.logger.info("Variables loaded from pretrained network\n{}".format(vars_info_vl(initialized_variables)))
        self.logger.info("Model building ends")

    def generate_sess(self):
        try: self.sess
        except AttributeError:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess=tf.Session(config=config)

    def set_up_train(self):
        self.logger.info("Model setting up train starts")

        decay_func = DECAY_DICT[self.args.dtype]
        if hasattr(self, 'start_epoch'):
            self.logger.info("Current start epoch : {}".format(self.start_epoch))
            DECAY_PARAMS_DICT[self.args.hdtype][self.args.nbatch][self.args.hdptype]['initial_step'] = self.nbatch_train*self.start_epoch
        self.lr, update_step_op = decay_func(**DECAY_PARAMS_DICT[self.args.dtype][self.args.nbatch][self.args.dptype])

        print(vars_info_vl(tf.trainable_variables()))
        update_ops = tf.get_collection("update_ops")
        with tf.control_dependencies(update_ops+[update_step_op]):
            self.train_op = get_multi_train_op(tf.train.AdamOptimizer, self.loss, [self.lr], [tf.trainable_variables()])

        self.graph_ops_dict = {'train' : [self.train_op, self.loss], 'val' : self.loss, 'test' : self.loss}
        
        self.val_embed_tensor1 = tf.placeholder(tf.float32, shape=[self.args.nbatch, self.args.m])
        self.val_embed_tensor2 = tf.placeholder(tf.float32, shape=[self.nval, self.args.m])

        self.p_dist = math_ops.add(
                    math_ops.reduce_sum(math_ops.square(self.val_embed_tensor1), axis=[1], keep_dims=True),
                    math_ops.reduce_sum(math_ops.square(array_ops.transpose(self.val_embed_tensor2)), axis=[0], keep_dims=True))-\
                2.0 * math_ops.matmul(self.val_embed_tensor1, array_ops.transpose(self.val_embed_tensor2)) # [batch_size, 1], [1, ndata],  [batch_size, ndata]

        self.p_dist = math_ops.maximum(self.p_dist, 0.0) # [batch_size, ndata] 
        self.p_dist = math_ops.multiply(self.p_dist, math_ops.to_float(math_ops.logical_not(math_ops.less_equal(self.p_dist, 0.0))))
        self.p_max_idx = tf.nn.top_k(-self.p_dist, k=2)[1] # [batch_size, 2] # get smallest 2

        self.logger.info("Model setting up train ends")

    def initialize(self):
        '''Initialize uninitialized variables'''
        self.logger.info("Model initialization starts")
        try:
            self.sess
        except NameError:
            self.sess=tf.Session()
        rest_initializer(self.sess) 
        if not hasattr(self, 'start_epoch'):
            self.start_epoch = 0
        self.logger.info("Model initialization ends")

    def save(self, global_step, save_dir):
        self.logger.info("Model save starts")
        for f in glob.glob(save_dir+'*'):
            os.remove(f)
        saver=tf.train.Saver(max_to_keep = 5)
        saver.save(self.sess, os.path.join(save_dir, 'model'), global_step = global_step)
        self.logger.info("Model save in %s"%save_dir)
        self.logger.info("Model save ends")

    def restore(self, save_dir):
        """Restore all variables in graph with the latest version"""
        self.logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        self.logger.info("Restoring from {}".format(latest_checkpoint))
        self.start_epoch = int(os.path.basename(latest_checkpoint)[len('model')+1:])
        try:
            self.sess
        except NameError:
            self.sess= tf.Session()
        saver.restore(self.sess, latest_checkpoint)
        self.logger.info("Restoring model done.")        

    def run_batch(self, key='train'):
        '''
        self.args :
            key - string
                train, test, val
        Return : 
            following graph operations
        '''
        assert key in ['train', 'test', 'val'], "key should be train or val or test"
        if self.args.ltype=='npair':
            batch_anc_img, batch_pos_img, batch_anc_label, batch_pos_label = self.dataset_dict[key].next_batch(batch_size=self.args.nbatch)
 
            feed_dict = {
                        self.anc_img : batch_anc_img,\
                        self.pos_img : batch_pos_img,\
                        self.label : batch_anc_label,\
                        self.istrain : True if key in ['train'] else False
                        }
            return self.sess.run(self.graph_ops_dict[key], feed_dict=feed_dict)
        else:# triplet
            batch_img, batch_label = self.dataset_dict[key].next_batch(batch_size=self.args.nbatch)
            feed_dict = {
                        self.img : batch_img,\
                        self.label : batch_label,\
                        self.istrain : True if key in ['train'] else False
                         }
            return self.sess.run(self.graph_ops_dict[key], feed_dict=feed_dict)

    def train(self, epoch, save_dir, board_dir):
        self.logger.info("Model training starts")

        self.train_writer = SummaryWriter(board_dir+'train') 
        self.val_writer = SummaryWriter(board_dir+'val') 
        
        max_val_recall = -1
        self.logger.info("Current epoch : {}/{}".format(self.start_epoch, epoch))
        self.logger.info("Current lr : {}".format(self.sess.run(self.lr)))

        if self.args.ltype=='npair':
            def custom_apply_tf_op(inputs, output_gate):
                return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.anc_img, output_gate=output_gate, batch_size=self.args.nbatch//2, dim=4, train_gate=self.istrain)

        else: # triplet
            def custom_apply_tf_op(inputs, output_gate):
                return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.img, output_gate=output_gate, batch_size=self.args.nbatch, dim=4, train_gate=self.istrain)

        for epoch_ in range(self.start_epoch, epoch):

            train_epoch_loss = 0
            for _ in tqdm(range(self.nbatch_train), ascii = True, desc="batch"):
                _, batch_loss = self.run_batch(key='train')
                train_epoch_loss += batch_loss
 
            # averaging
            train_epoch_loss /= self.nbatch_train
 
            if self.args.ltype=='npair':
                self.val_embed = custom_apply_tf_op(inputs=self.val_image, output_gate=self.anc_embed)
            else: # triplet
                self.val_embed = custom_apply_tf_op(inputs=self.val_image, output_gate=self.embed)

            val_p1 = get_recall_at_1_efficient(
                            data=self.val_embed, label=self.val_label,\
                            input1_tensor=self.val_embed_tensor1, input2_tensor=self.val_embed_tensor2,\
                            idx_tensor=self.p_max_idx, session=self.sess)


            self.logger.info("Epoch({}/{}) train loss = {} val p@1 = {}"\
                    .format(epoch_ + 1, epoch, train_epoch_loss, val_p1))

            if train_epoch_loss!=train_epoch_loss:# if nan
                break

            self.train_writer.add_summary("loss", train_epoch_loss, epoch_)
            self.train_writer.add_summary("learning rate", self.sess.run(self.lr), epoch_)
            self.val_writer.add_summary("p@1", val_p1, epoch_)
  
            if epoch_ == self.start_epoch or max_val_recall < val_p1:
                max_val_recall = val_p1
                self.save(epoch_+1, save_dir)
 
        self.logger.info("Model training ends")

    def regen_session(self):
        tf.reset_default_graph()
        self.logger.info("Regenerate sessions")
        self.sess.close()
        self.sess = tf.Session()

    def prepare_test(self):
        self.logger.info("Model preparing test")

        if self.args.ltype=='npair':
            def custom_apply_tf_op(inputs, output_gate):
                return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.anc_img, output_gate=output_gate, batch_size=self.args.nbatch//2, dim=4, train_gate=self.istrain)
            self.test_embed = custom_apply_tf_op(inputs=self.test_image, output_gate=self.anc_embed)

        else: # triplet
            def custom_apply_tf_op(inputs, output_gate):
                return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.img, output_gate=output_gate, batch_size=self.args.nbatch, dim=4, train_gate=self.istrain)
            self.test_embed = custom_apply_tf_op(inputs=self.test_image, output_gate=self.embed)

    def test_metric(self, k_set):
        self.logger.info("Model testing metric starts")
        if not hasattr(self, 'te_te_distance'):
            self.regen_session()
            self.te_te_distance = pairwise_distance_euclid_efficient(input1=self.test_embed, input2=self.test_embed, session=self.sess, batch_size=128)
            self.logger.info("Calculating pairwise distance from test embeddings")
        performance = evaluate_metric_te(test_label=self.test_label, te_te_distance=self.te_te_distance, k_set=k_set, logger=self.logger) 
        return performance

    def test_th(self, activate_k, k_set):
        self.logger.info("Model testing thresholding starts")
        self.logger.info("Activation k(={}) in embeddings(={})".format(activate_k, self.args.m))
        self.regen_session()
        test_k_activate = activate_k_2D(self.test_embed, k=activate_k, session=self.sess) # [ntest, args.m]

        if not hasattr(self, 'te_te_distance'):
            self.regen_session()
            self.te_te_distance = pairwise_distance_euclid_efficient(input1=self.test_embed, input2=self.test_embed, session=self.sess, batch_size=128)
            self.logger.info("Calculating pairwise distance from test embeddings")

        performance = evaluate_hash_te(test_hash_key=test_k_activate, te_te_distance=self.te_te_distance,\
                                          te_te_query_key=test_k_activate, te_te_query_value=self.test_embed,\
                                          test_label=self.test_label, ncls_test=self.ncls_test,\
                                          activate_k=activate_k, k_set=k_set, logger=self.logger)

        self.logger.info("Model testing thresholding ends")
        return performance

    def test_vq(self, activate_k, k_set):
        self.logger.info("Model testing vq starts")
        self.logger.info("Activation k(={}) in buckets(={})".format(activate_k, self.args.m))
        if not hasattr(self, 'test_kmc'):
            self.regen_session()
            self.test_kmc = KMeansClustering(self.test_embed, self.args.m) 
        if not hasattr(self, 'te_te_distance'):
            self.regen_session()
            self.te_te_distance = pairwise_distance_euclid_efficient(input1=self.test_embed, input2=self.test_embed, session=self.sess, batch_size=128)
            self.logger.info("Calculating pairwise distance from test embeddings")

        te_te_query_value = self.test_kmc.k_hash(self.test_embed, self.sess) # [ntest, args.d] center test
        te_te_query_key = activate_k_2D(te_te_query_value, k=activate_k, session=self.sess) # [ntest, args.d]
        test_hash_key = te_te_query_key
        self.regen_session()

        performance = evaluate_hash_te(test_hash_key=test_hash_key, te_te_distance=self.te_te_distance,\
                                          te_te_query_key=te_te_query_key, te_te_query_value=te_te_query_value,\
                                          test_label=self.test_label, ncls_test=self.ncls_test,\
                                          activate_k=activate_k, k_set=k_set, logger=self.logger)

        self.logger.info("Model testing vq ends")
        return performance

    def delete(self):
        tf.reset_default_graph()
        self.logger.remove()
        del self.logger

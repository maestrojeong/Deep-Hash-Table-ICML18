import sys
sys.path.append('../../configs')
sys.path.append('../../utils')
sys.path.append('../../tfops')
# configs
from info import CIFARNCLASS

# ../../utils 
from evaluation import evaluate_hash_te_tr
from ortools_op import SolveMaxMatching
from logger_op import LoggerManager
from eval_op import get_nmi_suf_quick
from gpu_op import selectGpuById
from np_op import activate_k_2D

# ../../tfops
from ops import rest_initializer, vars_info_vl, get_multi_train_op, get_initialized_vars
from nets import CONV_DICT
from dist import pairwise_distance_euclid, pairwise_distance_euclid_v2
from hash_dist import triplet_semihard_loss_hash, npairs_loss_hash,\
                      PAIRWISE_DISTANCE_WITH_OBJECTIVE_DICT, PAIRWISE_SIMILARITY_WITH_OBJECTIVE_DICT
from transform_op import apply_tf_op, pairwise_distance_euclid_efficient
from summary_op import SummaryWriter
from lr_op import DECAY_DICT

# ./
from local_config import DECAY_PARAMS_DICT

from tqdm import tqdm
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import glob
import os

import logging
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S" )

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
        self.train_image = self.dataset_dict['train'].image
        self.train_label = self.dataset_dict['train'].label
        self.ntrain, self.height, self.width, self.nchannel = self.train_image.shape
        self.ncls_train = self.train_dataset.nclass
        self.nbatch_train = self.ntrain//self.args.nbatch
        self.logger.info("Setting train_dataset ends")

    def set_test_dataset(self, test_dataset):
        self.logger.info("Setting test_dataset starts")
        self.test_dataset = test_dataset
        self.dataset_dict['test'] = self.test_dataset
        self.test_image = self.dataset_dict['test'].image 
        self.test_label = self.dataset_dict['test'].label
        self.ncls_test = self.test_dataset.nclass
        self.ntest = self.test_dataset.ndata
        self.nbatch_test = self.ntest//self.args.nbatch
        self.logger.info("Setting test_dataset ends")

    def set_val_dataset(self, val_dataset):
        self.logger.info("Setting val_dataset starts")
        self.val_dataset = val_dataset
        self.dataset_dict['val'] = self.val_dataset
        self.val_image = self.dataset_dict['val'].image 
        self.val_label = self.dataset_dict['val'].label
        self.nval = self.val_dataset.ndata
        self.ncls_val = self.val_dataset.nclass
        self.nbatch_val = self.nval//self.args.nbatch
        self.logger.info("Setting val_dataset ends")
    
    def switch_log_path(self, logfilepath):
        self.logger.remove()
        print("Log file switched from {} to {}".format(self.logfilepath, logfilepath))
        self.logfilepath = logfilepath
        self.logger = LoggerManager(self.logfilepath, __name__)

    def build(self, pretrain=False):
        self.logger.info("Model building starts")
        tf.reset_default_graph()
        if self.args.hltype == 'npair':
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
        if self.args.hltype == 'npair':
            self.anc_last, _ = self.conv_net(self.anc_img, is_training=self.istrain, reuse=False)
            self.pos_last, _ = self.conv_net(self.pos_img, is_training=self.istrain, reuse=True)
            self.anc_last = tf.nn.relu(self.anc_last)
            self.pos_last = tf.nn.relu(self.pos_last)
        else:
            self.last, _ = self.conv_net(self.img, is_training=self.istrain, reuse=False)
            self.last = tf.nn.relu(self.last)

        with slim.arg_scope([slim.fully_connected], activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0005), biases_initializer=tf.zeros_initializer()):
            if self.args.hltype == 'npair':
                with tf.variable_scope('Embed', reuse=False): self.anc_embed = slim.fully_connected(self.anc_last, self.args.m, scope="fc1")
                with tf.variable_scope('Embed', reuse=True): self.pos_embed = slim.fully_connected(self.pos_last, self.args.m, scope="fc1")
            else:#triplet
                with tf.variable_scope('Embed', reuse=False): self.embed = slim.fully_connected(self.last, self.args.m, scope = "fc1")

        initialized_variables=get_initialized_vars(self.sess)
        self.logger.info("Variables loaded from pretrained network\n{}".format(vars_info_vl(initialized_variables)))
        self.logger.info("Model building ends")
    
    def build_hash(self): 
        self.logger.info("Model building train hash starts")

        self.mcf = SolveMaxMatching(nworkers=self.args.nsclass, ntasks=self.args.d, k=self.args.k, pairwise_lamb=self.args.plamb)

        if self.args.hltype=='triplet': self.objective = tf.placeholder(tf.float32, shape=[self.args.nbatch, self.args.d])
        else: self.objective = tf.placeholder(tf.float32, shape=[self.args.nbatch//2, self.args.d])

        with slim.arg_scope([slim.fully_connected], activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0005), biases_initializer=tf.zeros_initializer()):
            if self.args.hltype=='triplet':
                self.embed_k_hash = self.last
                with tf.variable_scope('Hash', reuse=False): self.embed_k_hash = slim.fully_connected(self.embed_k_hash, self.args.d, scope="fc1")
                self.embed_k_hash_l2_norm = tf.nn.l2_normalize(self.embed_k_hash, dim=-1) # embedding with l2 normalization
                self.pairwise_distance =PAIRWISE_DISTANCE_WITH_OBJECTIVE_DICT[self.args.hdt]
                self.loss_hash = triplet_semihard_loss_hash(labels=self.label, embeddings=self.embed_k_hash_l2_norm, objectives=self.objective,\
                                pairwise_distance=self.pairwise_distance, margin=self.args.hma)
            else: 
                self.anc_embed_k_hash = self.anc_last
                self.pos_embed_k_hash = self.pos_last
                with tf.variable_scope('Hash', reuse=False): self.anc_embed_k_hash = slim.fully_connected(self.anc_embed_k_hash, self.args.d, scope="fc1")
                with tf.variable_scope('Hash', reuse=True): self.pos_embed_k_hash = slim.fully_connected(self.pos_embed_k_hash, self.args.d, scope="fc1")
                self.similarity_func =PAIRWISE_SIMILARITY_WITH_OBJECTIVE_DICT[self.args.hdt]
                self.loss_hash = npairs_loss_hash(labels=self.label, embeddings_anchor=self.anc_embed_k_hash, embeddings_positive=self.pos_embed_k_hash,\
                        objective=self.objective, similarity_func=self.similarity_func, reg_lambda=self.args.hlamb)

        self.logger.info("Model building train hash ends")

    def set_up_train_hash(self):
        self.logger.info("Model setting up train hash starts")

        decay_func = DECAY_DICT[self.args.hdtype]
        if hasattr(self, 'start_epoch'):
            self.logger.info("Current start epoch : {}".format(self.start_epoch))
            DECAY_PARAMS_DICT[self.args.hdtype][self.args.nbatch][self.args.hdptype]['initial_step'] = self.nbatch_train*self.start_epoch
        self.lr_hash, update_step_op = decay_func(**DECAY_PARAMS_DICT[self.args.hdtype][self.args.nbatch][self.args.hdptype])

        update_ops = tf.get_collection("update_ops")

        var_slow_list, var_fast_list = list(), list()
        for var in tf.trainable_variables():
            if 'Hash' in var.name: var_fast_list.append(var)
            else: var_slow_list.append(var)

        with tf.control_dependencies(update_ops+[update_step_op]):
            self.train_op_hash = get_multi_train_op(tf.train.AdamOptimizer, self.loss_hash, [0.1*self.lr_hash, self.lr_hash], [var_slow_list, var_fast_list])

        self.EMBED_HASH = self.anc_embed_k_hash if self.args.hltype=='npair' else self.embed_k_hash
        self.max_k_idx = tf.nn.top_k(self.EMBED_HASH, k=self.args.k)[1] # [batch_size, k]

        self.graph_ops_hash_dict = {'train' : [self.train_op_hash, self.loss_hash], 'val' : self.loss_hash}
        self.logger.info("Model setting up train hash ends")

    def generate_sess(self):
        try: self.sess
        except AttributeError:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess=tf.Session(config=config)

    def initialize(self):
        '''Initialize uninitialized variables'''
        self.logger.info("Model initialization starts")
        self.generate_sess()
        rest_initializer(self.sess) 
        self.start_epoch = 0
        val_p_dist = pairwise_distance_euclid_efficient(input1=self.val_embed, input2=self.val_embed, session=self.sess, batch_size=self.args.nbatch)
        self.logger.info("Calculating pairwise distance of validation data")
        self.val_arg_sort = np.argsort(val_p_dist, axis=1)
        self.logger.info("Model initialization ends")

    def save(self, global_step, save_dir):
        self.logger.info("Model save starts")
        for f in glob.glob(save_dir+'*'): os.remove(f)
        saver=tf.train.Saver(max_to_keep = 5)
        saver.save(self.sess, os.path.join(save_dir, 'model'), global_step = global_step)
        self.logger.info("Model save in %s"%save_dir)
        self.logger.info("Model save ends")

    def save_hash(self, global_step, save_dir):
        self.logger.info("Model save starts")
        for f in glob.glob(save_dir+'*'): os.remove(f)
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
        self.generate_sess()
        saver.restore(self.sess, latest_checkpoint)
        self.logger.info("Restoring model done.")        

    def restore_hash(self, save_dir):
        """Restore all variables in graph with the latest version"""
        self.logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        self.logger.info("Restoring from {}".format(latest_checkpoint))
        self.start_epoch = int(os.path.basename(latest_checkpoint)[len('model')+1:])
        self.generate_sess()
        saver.restore(self.sess, latest_checkpoint)
        self.logger.info("Restoring model done.")        

    def run_batch_hash(self, key='train'):
        '''
        self.args :
            key - string
                    train, test, val
        Return : 
            following graph operations
        '''
        assert key in ['train', 'test', 'val'], "key should be train or val or test"
        if self.args.hltype=='npair':
            batch_anc_img, batch_pos_img, batch_anc_label, batch_pos_label = self.dataset_dict[key].next_batch(batch_size=self.args.nbatch)
 
            feed_dict = {
                        self.anc_img : batch_anc_img,\
                        self.pos_img : batch_pos_img,\
                        self.label : batch_anc_label,\
                        self.istrain : True if key in ['train'] else False
                        }
            
            # [self.args.nbatch//2, self.args.d]
            anc_unary, pos_unary = self.sess.run([self.anc_embed_k_hash, self.pos_embed_k_hash], feed_dict=feed_dict)

            unary = 0.5*(anc_unary+pos_unary) # [batch_size//2, d]
            unary = np.mean(np.reshape(unary, [self.args.nsclass, -1, self.args.d]), axis=1) # [nsclass, d]

            results = self.mcf.solve(unary)
            objective = np.zeros([self.args.nsclass, self.args.d], dtype=np.float32) # [nsclass, d]
            for i, j in results: objective[i][j]=1
            objective = np.reshape(np.transpose(np.tile(np.transpose(objective, [1,0]), [self.args.nbatch//(2*self.args.nsclass), 1]), [1,0]), [self.args.nbatch//2, self.args.d]) # [batch_size//2, d]
            feed_dict[self.objective] = objective
            return self.sess.run(self.graph_ops_hash_dict[key], feed_dict=feed_dict)
        else:
            batch_img, batch_label = self.dataset_dict[key].next_batch(batch_size=self.args.nbatch)
            feed_dict = {
                         self.img : batch_img,\
                         self.label : batch_label,\
                         self.istrain : True if key in ['train'] else False
                         }

            unary = self.sess.run(self.embed_k_hash_l2_norm, feed_dict=feed_dict) # [nsclass, d]
            unary = np.mean(np.reshape(unary, [self.args.nsclass, -1, self.args.d]), axis=1) # [nsclass, d]

            results = self.mcf.solve(unary)
            objective = np.zeros([self.args.nsclass, self.args.d], dtype=np.float32)
            for i, j in results: objective[i][j]=1

            objective = np.reshape(np.transpose(np.tile(np.transpose(objective, [1,0]), [self.args.nbatch//self.args.nsclass , 1]), [1,0]), [self.args.nbatch, -1]) # [batch_size, d]
            feed_dict[self.objective] = objective
            return self.sess.run(self.graph_ops_hash_dict[key], feed_dict=feed_dict) 

    def train_hash(self, epoch, save_dir, board_dir):
        self.logger.info("Model training starts")

        self.train_writer_hash = SummaryWriter(board_dir+'train') 
        self.val_writer_hash = SummaryWriter(board_dir+'val') 

        self.logger.info("Current epoch : {}/{}".format(self.start_epoch, epoch))
        self.logger.info("Current lr : {}".format(self.sess.run(self.lr_hash)))

        if self.args.hltype=='npair':
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.anc_img, output_gate=output_gate, batch_size=self.args.nbatch//2, dim=4, train_gate=self.istrain)
        else: # triplet
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.img, output_gate=output_gate, batch_size=self.args.nbatch, dim=4, train_gate=self.istrain)

        val_max_k_idx = custom_apply_tf_op(inputs=self.val_image, output_gate=self.max_k_idx)
        val_nmi, val_suf = get_nmi_suf_quick(index_array=val_max_k_idx, label_array=self.val_label, ncluster=self.args.d, nlabel=self.ncls_val)
        nsuccess=0
        for i in range(self.nval):
            for j in self.val_arg_sort[i]:
                if i==j: continue
                if len(set(val_max_k_idx[j])&set(val_max_k_idx[i]))>0:
                    if self.val_label[i]==self.val_label[j]: nsuccess+=1
                    break
        val_p1 = nsuccess/self.nval
        max_val_p1=val_p1

        self.val_writer_hash.add_summary("suf", val_suf, self.start_epoch)
        self.val_writer_hash.add_summary("nmi", val_nmi, self.start_epoch)
        self.val_writer_hash.add_summary("p1", val_p1, self.start_epoch)

        for epoch_ in range(self.start_epoch, self.start_epoch+epoch):
            train_epoch_loss = 0
            val_epoch_loss = 0
            for _ in tqdm(range(self.nbatch_train), ascii = True, desc="batch"):
                _, batch_loss = self.run_batch_hash(key='train')
                train_epoch_loss += batch_loss	

            val_max_k_idx = custom_apply_tf_op(inputs=self.val_image, output_gate=self.max_k_idx)
            val_nmi, val_suf = get_nmi_suf_quick(index_array=val_max_k_idx, label_array=self.val_label, ncluster=self.args.d, nlabel=self.ncls_val)
            nsuccess=0
            for i in range(self.nval):
                for j in self.val_arg_sort[i]:
                    if i==j: continue
                    if len(set(val_max_k_idx[j])&set(val_max_k_idx[i]))>0:
                        if self.val_label[i]==self.val_label[j]: nsuccess+=1
                        break
            val_p1 = nsuccess/self.nval
            # averaging
            train_epoch_loss /= self.nbatch_train

            self.logger.info("Epoch({}/{}) train loss = {} val suf = {} val nmi = {} val p1 = {}"\
                    .format(epoch_ + 1, epoch, train_epoch_loss, val_suf, val_nmi, val_p1))	

            self.train_writer_hash.add_summary("loss", train_epoch_loss, epoch_+1)
            self.train_writer_hash.add_summary("learning rate", self.sess.run(self.lr_hash), epoch_+1)
            self.val_writer_hash.add_summary("suf", val_suf, epoch_+1)
            self.val_writer_hash.add_summary("nmi", val_nmi, epoch_+1)
            self.val_writer_hash.add_summary("p1", val_p1, epoch_+1)

            if epoch_ == self.start_epoch or max_val_p1 < val_p1:
                max_val_p1 = val_p1
                self.save_hash(epoch_+1, save_dir)
        self.logger.info("Model training ends")

    def regen_session(self):
        tf.reset_default_graph()
        self.sess.close()
        self.sess = tf.Session()

    def prepare_test(self):
        self.logger.info("Model preparing test")
        if self.args.hltype=='npair':
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.anc_img, output_gate=output_gate, batch_size=self.args.nbatch//2, dim=4, train_gate=self.istrain)
            self.val_embed = custom_apply_tf_op(inputs=self.val_image, output_gate=self.anc_embed)
            self.test_embed = custom_apply_tf_op(inputs=self.test_image, output_gate=self.anc_embed)
            self.train_embed = custom_apply_tf_op(inputs=self.train_image, output_gate=self.anc_embed)

        else: # triplet
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.img, output_gate=output_gate, batch_size=self.args.nbatch, dim=4, train_gate=self.istrain)
            self.val_embed = custom_apply_tf_op(inputs=self.val_image, output_gate=self.embed)
            self.test_embed = custom_apply_tf_op(inputs=self.test_image, output_gate=self.embed)
            self.train_embed = custom_apply_tf_op(inputs=self.train_image, output_gate=self.embed)

    def prepare_test_hash(self):
        self.logger.info("Model preparing test")
        if self.args.hltype=='npair':
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.anc_img, output_gate=output_gate, batch_size=self.args.nbatch//2, dim=4, train_gate=self.istrain)
        else: # triplet
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.img, output_gate=output_gate, batch_size=self.args.nbatch, dim=4, train_gate=self.istrain)
        self.test_k_hash = custom_apply_tf_op(inputs=self.test_image, output_gate=self.EMBED_HASH)
        self.train_k_hash = custom_apply_tf_op(inputs=self.train_image, output_gate=self.EMBED_HASH)

    def test_hash_metric(self, activate_k, k_set):
        self.logger.info("Model testing k hash starts")
        self.logger.info("Activation k(={}) in buckets(={})".format(activate_k, self.args.d))
        if not hasattr(self, 'te_tr_distance') and not hasattr(self, 'te_te_distance'):
            self.regen_session()
            self.te_tr_distance = self.sess.run(pairwise_distance_euclid_v2(tf.convert_to_tensor(self.test_embed, dtype=tf.float32), tf.convert_to_tensor(self.train_embed, dtype=tf.float32)))
            self.te_te_distance = self.sess.run(pairwise_distance_euclid(tf.convert_to_tensor(self.test_embed, dtype=tf.float32)))

        test_k_activate = activate_k_2D(self.test_k_hash, k=activate_k, session=self.sess) # [ntest, args.d]
        train_k_activate = activate_k_2D(self.train_k_hash, k=activate_k, session=self.sess) # [ntrain, args.d]
        self.regen_session()

        performance = evaluate_hash_te_tr(train_hash_key=train_k_activate, test_hash_key=test_k_activate,\
                                            te_tr_distance=self.te_tr_distance, te_te_distance=self.te_te_distance,\
                                            te_tr_query_key=test_k_activate, te_tr_query_value=self.test_k_hash,\
                                            te_te_query_key=test_k_activate, te_te_query_value=self.test_k_hash,\
                                            train_label=self.train_label, test_label=self.test_label,\
                                            ncls_train=self.ncls_train, ncls_test=self.ncls_test,\
                                            activate_k=activate_k, k_set=k_set, logger=self.logger)

        self.logger.info("Model testing k hash ends")
        return performance

    def delete(self):
        tf.reset_default_graph()
        self.logger.remove()
        del self.logger

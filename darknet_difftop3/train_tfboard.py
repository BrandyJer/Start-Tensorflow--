# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:54:02 2017

@author: linjian_sx
"""
"""
jiezhen change:
    line22 => 24
    line32 => 33
    line40 => 41
    line63 => 64
    line81 => 82
    line85 => 86
"""
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import os
import random
import time
from PIL import Image
from decode_tools import decode_from_tfrecords
#from zf_net import tiny_darknet
from net import tiny_darknet

max_iters = 80000
#for code test
#max_iters = 100

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def train(is_ft=True):
    with tf.Graph().as_default():
        with tf.variable_scope("model") as scope:
#            train_queue = ["train_data2.tfrecords"]
            train_queue = ["train_quarter.tfrecords"]
            images, labels = decode_from_tfrecords(train_queue,128)
            logits = tiny_darknet(images)
            tf.summary.image('iuput', images)
#            logits = tf.nn.softmax(tf.reduce_mean(logits,[1,2]))
            logits = tf.reduce_mean(logits,[1,2])
            print logits.get_shape().as_list()
            loss =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            reg_loss = tf.add_n(tf.losses.get_regularization_losses())
#            with tf.name_scope('total_loss'):
            total_loss = tf.reduce_mean(loss)+reg_loss
            tf.summary.scalar('total_loss', total_loss)

            opt = tf.train.MomentumOptimizer(0.01,0.9)
            global_step = tf.Variable(0, name='global_step', trainable=False)

#            learning_rate = tf.train.exponential_decay(0.1, global_step, 10200, 0.35, staircase=True)
#            min_lr= tf.constant(0.00001, name='min_lr')
#            if learning_rate<0.00001:
#                learning_rate=0.00001
#            learning_rate = tf.Session.run(tf.where(tf.greater(min_lr, learning_rate), min_lr, learning_rate))
#            opt = tf.train.MomentumOptimizer(learning_rate,0.9)
#            opt.minimize(total_loss, global_step=global_step)

            train_op = slim.learning.create_train_op(total_loss, opt, global_step=global_step)


            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                total_loss = control_flow_ops.with_dependencies([updates], total_loss)
            
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)
            init = tf.initialize_all_variables()
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            sess = tf.Session()
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('/root/linjian/darknet_0/models/lr0.01_iter30w_qnew/lr0.01', sess.graph)
            sess.run(init)

            tf.train.start_queue_runners(sess=sess)

            if is_ft:#if not train model
#                model_file=tf.train.latest_checkpoint('./model_max')
                model_file=tf.train.latest_checkpoint('./models/lr0.01_iter30w_qnew')
                saver.restore(sess, model_file)
#            if learning_rate<0.00001:
#                learning_rate=0.00001
            #is_ft = False
#            ckpt = tf.train.get_checkpoint_state('./models')
#            if ckpt and ckpt.model_checkpoint_path:
#                model_file=tf.train.latest_checkpoint('./models')
#                saver.restore(sess, model_file)
            tf.logging.set_verbosity(tf.logging.INFO)    
            loss_cnt = 0.0
            loss_flag = 999.0
            for step in range(max_iters):
                _, loss_value = sess.run([train_op, total_loss])
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                loss_cnt+=loss_value
                if step % 10 == 0:
                    format_str = ('%s: step %d, loss = %.4f')
                    if step == 0:
                        avg_loss_cnt = loss_cnt
                    else:
                        avg_loss_cnt = loss_cnt/10.0
                    summary_str = sess.run(merged)
                    train_writer.add_summary(summary_str, step)
                    print(format_str % (datetime.now(), step, avg_loss_cnt))
                    loss_cnt = 0.0
                if step % 4000 == 0 or (step + 1) == max_iters:
#                if step % 50 == 0 or (step + 1) == max_iters:
#                    checkpoint_path = os.path.join('/root/classify/model', 'dp15_model.ckpt')
                    checkpoint_path = os.path.join('/root/linjian/darknet_0/models/lr0.01_iter30w_qnew/lr0.01', 'model.ckpt')#save model path
                    saver.save(sess, checkpoint_path, global_step=step)
            train_writer.close()





train()

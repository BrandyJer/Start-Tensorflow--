# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:54:02 2017

@author: linjian_sx
jiezhen change:
    line24 => 25
    line44 => 45
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
from net import tiny_darknet
from decode_tools import decode_from_tfrecords_eval


#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval():
    with tf.Graph().as_default():
        with tf.variable_scope("model") as scope:
            thre = 0.9          
#           ###################################################
            eval_loss_ = tf.placeholder(tf.float32)
            eval_acc_ = tf.placeholder(tf.float32)
            eval_pre_ = tf.placeholder(tf.float32)
            eval_recall_ = tf.placeholder(tf.float32)

            tf.summary.scalar('eval_acc', eval_acc_)
            tf.summary.scalar('eval_loss', eval_loss_)
            tf.summary.scalar('eval_pre', eval_pre_)
            tf.summary.scalar('eval_recall', eval_recall_)
            
            merged = tf.summary.merge_all()
#            train_writer = tf.summary.FileWriter('/root/linjian/darknet_0/eval/dp0.3_lr0.001/lr0.0003', sess.graph)
#           ###################################################
            eval_queue = "./tf_data/validation_quarter.tfrecords"
            images_eval,labels_eval = decode_from_tfrecords_eval(eval_queue,1)
            logits_eval = tiny_darknet(images_eval,False)
            logits_eval = tf.reduce_mean(logits_eval,[1,2])
            loss_eval =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_eval, logits=logits_eval)
            logits_eval = tf.nn.sigmoid(logits_eval) 
            saver = tf.train.Saver(tf.all_variables())
            init = tf.initialize_all_variables()
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            sess = tf.Session()
            train_writer = tf.summary.FileWriter('/root/linjian/darknet_0/models/try-linjian/all_data/0_wd4e5-0.15/eval-0.9', sess.graph)
            sess.run(init)

            tf.train.start_queue_runners(sess=sess)
                
#            model_file=tf.train.latest_checkpoint('./model_max')
#            model_file=tf.train.latest_checkpoint('/root/jiezhen/Code/model_max')
#            model_file='/root/jiezhen/Code/model_max/model.ckpt-11800.data-00000-of-00001'
#            model_file='/root/JZ_test/darknet0_model/model.ckpt-9800.data-00000-of-00001'
#            saver.restore(sess,model_file)
#            saver = tf.train.import_meta_graph('/root/JZ_test/darknet0_model/model.ckpt-9800.meta')
#            saver.restore(sess, os.path.join('/root/JZ_test/darknet0_model', 'model.ckpt'))
#            ckpt = tf.train.get_checkpoint_state("/root/JZ_test/darknet0_model")
            ckpt = tf.train.get_checkpoint_state("/root/linjian/darknet_0/models/try-linjian/all_data/0_wd4e5-0.15")
#####            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
#####            step = 0 
            for model in range(-100, 0):    #-55==>model-2000
#####            step+=1
                saver.restore(sess, ckpt.all_model_checkpoint_paths[model])
                num = 4172
                mse = 0
                cnt = 0
                recall = 0
                acc = 0
                for eval_iter in range(num):
                    loss_value_eval,l,gt = sess.run([loss_eval,logits_eval,labels_eval])
                    mse+=loss_value_eval
                    if l[0][0]>= thre:
                        predict = 0
                    else:
                        predict = 1
                    if predict == 0:
                        cnt+=1
                        if gt == 0:
                            recall+=1
                    if predict == gt:
                        acc+=1
                
#####                print("The "+str(step)+" iter eval loss:"+str(mse/float(num)))
                eval_loss = mse/float(num)
                eval_acc = acc/float(num)
                print("========Thre :"+str(thre)+'/'+str(model+100)+" _model eval ========")
                print("loss:"+str(eval_loss))
                print("acc:"+str(eval_acc))
                if cnt !=0:
                    eval_pre = recall/(float(cnt))
                    print('precision:'+str(eval_pre))
                else:
                    eval_pre = 0.0
                    print('precision:'+str(eval_pre))
                eval_recall = recall/(float(num/2))
                print('recall:'+str(eval_recall))
                print "================================="
                summary_str = sess.run(merged, feed_dict={eval_loss_: eval_loss[0],
                                                          eval_acc_: eval_acc,
                                                          eval_pre_: eval_pre,
                                                          eval_recall_: eval_recall})
                train_writer.add_summary(summary_str, model+100)
            train_writer.close()


eval()

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

def eval_val(Model_PATH, step):
    with tf.Graph().as_default():
        with tf.variable_scope("model") as scope:
            thre = 0.9          
            
            eval_queue = "validation_quarter.tfrecords"
            images_eval,labels_eval = decode_from_tfrecords_eval(eval_queue,1)
            logits_eval = tiny_darknet(images_eval,False)
            logits_eval = tf.reduce_mean(logits_eval,[1,2])
            loss_eval =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_eval, logits=logits_eval)
            logits_eval = tf.nn.sigmoid(logits_eval) 
            saver = tf.train.Saver(tf.all_variables())
            init = tf.initialize_all_variables()
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            sess.run(init)

            tf.train.start_queue_runners(sess=sess)
                
#            model_file=tf.train.latest_checkpoint('./model_max')
#            model_file=tf.train.latest_checkpoint('/root/jiezhen/Code/model_max')
#            model_file='/root/jiezhen/Code/model_max/model.ckpt-11800.data-00000-of-00001'
#            model_file='/root/JZ_test/darknet0_model/model.ckpt-9800.data-00000-of-00001'
#            saver.restore(sess,model_file)
#            saver = tf.train.import_meta_graph('/root/JZ_test/darknet0_model/model.ckpt-9800.meta')
#            saver.restore(sess, os.path.join('/root/JZ_test/darknet0_model', 'model.ckpt'))

#            ckpt = tf.train.get_checkpoint_state("./models/lr0.01_iter30w_qnew/lr0.01_wd4e5")
            ckpt = tf.train.get_checkpoint_state(Model_PATH)
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
#            step = 0 
#            while(True):
#                step+=1
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
                    
            print("The "+str(step)+" iter eval loss:"+str(mse/float(num)))
            print("acc:"+str(acc/float(num)))
            val0loss = mse/float(num)
            val0acc = acc/float(num)
            if cnt !=0:
                print('precision:'+str(recall/(float(cnt))))
                val0pre = recall/(float(cnt))
            print('recall:'+str(recall/(float(num/2))))
            val0recall = recall/(float(num/2))
            return val0loss, val0acc, val0pre, val0recall
#                time.sleep(600)

#eval()

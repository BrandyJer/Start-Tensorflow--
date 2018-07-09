"""
jiezhen change:
    line20 => 21
    line25 => 23
    line26 => 24
"""
import argparse
import time
import os
import numpy as np
import tensorflow as tf
#import tensorflow.contrib.slim as slim
from PIL import Image
from net import tiny_darknet
from decode_tools import decode_from_label_tfrecords



parser = argparse.ArgumentParser()
#parser.add_argument("--mode",default="cpu")
parser.add_argument("--mode",default="gpu")
parser.add_argument("--batchsize",default=16,type=int)
parser.add_argument("--tfdata",default='/root/jiezhen/Data_HP/HP_train/HP/valid.tfrecords')
parser.add_argument("--model_dir",default='/root/jiezhen/JZmodel')
#parser.add_argument("--tfdata",default='./valid.tfrecords')
#parser.add_argument("--model_dir",default='./model')

args = parser.parse_args()
error_flag = 0

if args.mode == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""#cpu version
elif args.mode == 'gpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"#gpu version with gpu_id=0
else:
    print("please type 'cpu' or 'gpu'")

if args.batchsize >= 256:
    print('batchsize do not be greater than 256')
    error_flag = 1


valid_data = args.tfdata
valid_queue = [valid_data]
batch_size = args.batchsize
model_dir = args.model_dir

if not os.path.exists(train_queue):
    print(train_queue+' don`t exists')
    error_flag = 1

if not os.path.exists(model_dir+'/checkpoint'):
    print(model_dir+'/checkpoint illegal')
    error_flag = 1

def time_benchmark():
    if error_flag:
        return 
    with tf.Graph().as_default():
        with tf.variable_scope("model") as scope:
            images, labels,name = decode_from_tfrecords(valid_queue,batch_size)         
            logits = tiny_darknet(images,False)
            logits = tf.reduce_mean(logits,[1,2])
        
            saver = tf.train.Saver(tf.all_variables())
            init = tf.initialize_all_variables()

            # Start running operations on the Graph.
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            sess.run(init)
       
         
            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
              
            index = 1
            num = int(12800/batch_size)
            while 1:
                model_file=tf.train.latest_checkpoint(args.model_dir)
                saver.restore(sess,model_file)
                start = time.clock()
                for step in range(num):
                    l = sess.run(logits)
                    print(step)
                elas = (time.clock() - start)/12800.0*1000
                print("time consume:"+str(elas))
                
                index+=1
                time.sleep(600)



time_benchmark()

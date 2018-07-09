import time
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image



def myscope(is_training=True,weight_decay=4e-5,stddev=0.1):
  batch_norm_params = {
      'is_training': is_training,
      'decay': 0.9,
#      'epsilon': 0.001,
      'updates_collections':None
  }
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  with slim.arg_scope([slim.conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,padding='SAME'):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer)as sc:
          return sc

def tiny_darknet(data,is_training=True):
    arg_scope=myscope(is_training)
    with slim.arg_scope(arg_scope):
        net = data*(1./255)-0.5
        net = slim.conv2d(net,16,[3,3],scope='conv0')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net,32, [3, 3],scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.conv2d(net,16, [1, 1],scope='conv4')
        net = slim.conv2d(net,128, [3, 3],scope='conv5')
        net = slim.conv2d(net,16, [1, 1],scope='conv6')
        net = slim.conv2d(net,128, [3, 3],scope='conv7')
        net = slim.max_pool2d(net, [2, 2], scope='pool8')
        net = slim.conv2d(net,32, [1, 1],scope='conv9')
        net = slim.conv2d(net,256, [3, 3],scope='conv10')
        net = slim.conv2d(net,32, [1, 1],scope='conv11')
        net = slim.conv2d(net,256, [3, 3],scope='conv12')
        net = slim.max_pool2d(net, [2, 2], scope='pool13')
        net = slim.conv2d(net,64, [1, 1],scope='conv14')
        net = slim.conv2d(net,512, [3, 3],scope='conv15')
        net = slim.conv2d(net,64, [1, 1],scope='conv16')
        net = slim.conv2d(net,512, [3, 3],scope='conv17')
#        net = slim.dropout(net, 0.3, scope='dropout17')
        net = slim.conv2d(net,128, [1, 1],scope='conv18')
#        net = slim.dropout(net, 0.5, scope='dropout18')
        net = slim.conv2d(net,2, [1, 1],scope='conv19')
        #net = tf.reduce_mean(net,[1,2])#scope='conv20')
        return net




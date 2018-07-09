# -*- coding: utf-8 -*-
import os, os.path
import random

f = open(r'/root/linjian/darknet_patch/raw_data/validation/val_list.txt',
         'r')
f_random = open(r'/root/linjian/darknet_patch/raw_data/validation/validation.txt',
         'a')

train_list = []
train_list = f.readlines()
random.shuffle(train_list)
for line in train_list:
    line = line.strip()+'\n'
    f_random.write(line)

f.close()
f_random.close()

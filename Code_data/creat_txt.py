# -*- coding: utf-8 -*-

import os, os.path

f = open(r'/root/linjian/darknet_patch/raw_data/validation/val_list.txt', 'a')
"""
path+'\t'+label
0:HP
1:YT
"""
path = r'/root/linjian/darknet_patch/raw_data/data'
HP = os.path.join(path, 'HP-0')
YT = os.path.join(path, 'YT-0')

for img in os.listdir(HP):
    img_path = os.path.join(HP, img)
    f.write(img_path+'\t'+'0'+'\n')

for img in os.listdir(YT):
    img_path = os.path.join(YT, img)
    f.write(img_path+'\t'+'1'+'\n')
f.close()


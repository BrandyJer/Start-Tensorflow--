# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 19:59:06 2018

@author: jiezhen_sx
"""

import os, os.path
import shutil
import random
#1 HP = 9YT:x-4,x-3..x,x+1..x+4
#截取的HP相邻距离小于9时：if 对应的原图不存在，则stillneed_YT += 1计算缺少多少张TY

HP_path = r"D:\jiezhen\workshop\Huaping_jz\JZ_patch\data\train\HP-2"
YT_path_sourse = r"D:\jiezhen\workshop\Huaping_jz\JZ_patch\data\Training Set\patch-new-2\images"
YT_path_end = r"D:\jiezhen\workshop\Huaping_jz\JZ_patch\data\train\YT-2" 

stillneed_YT = 0

for HP in os.listdir(HP_path):
    HPname = HP.split('.')[0]
    x = int(HPname)
    for i in range(x-4,x+5):
        YTimg_path = os.path.join(YT_path_sourse, str(i)+'.jpg')
        if os.path.exists(YTimg_path):
            shutil.move(YTimg_path, YT_path_end)
        else:
            stillneed_YT += 1 

#随机取stillneed_YT个YT
list_restYT = os.listdir(YT_path_sourse)
#从list中随机获取stillneed_YT个元素，作为一个片断返回 ,不改变原list
random_YT = random.sample(list_restYT, stillneed_YT)
for YT in random_YT:
    random_YT_path = os.path.join(YT_path_sourse, YT)
    shutil.move(random_YT_path, YT_path_end)
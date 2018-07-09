# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:10:33 2018

@author: jiezhen_sx
"""
import cv2
import os, os.path


def sliding_window(image):
    #slide a window across the image
    stride_h = 180
    stride_w = 256
    Img_H = image.shape[0]
    Img_W = image.shape[1]
    for h in range(0, Img_H, stride_h):
        for w in range(0, Img_W, stride_w):
            #yield the current window
            end_h = h+stride_h
            end_w = w+stride_w
            if end_h <= Img_H and end_w <= Img_W:
                yield image[h:end_h, w:end_w]
            else:
                star_h = h
                star_w = w
                if end_h > Img_H:
                    star_h = Img_H-stride_h
                if end_w > Img_W:
                    star_w = Img_W-stride_w
                yield image[star_h:star_h+stride_h, star_w:star_w+stride_w]
            
            
def output_patch(raw_image):
    output_list = []
    window_num = 0
    for window in sliding_window(raw_image):
        output_list.append(window)
        window_num = len(output_list)
    return output_list, window_num

###cut windsize patch form anysize image
#test_allwindow = []
#img_num = 0
#img_path = r'D:\jiezhen\workshop\Huaping_jz\JZ_patch\data\old_data\YT_validation'
#save_patch = r'D:\jiezhen\workshop\Huaping_jz\JZ_patch\data\old_data\Validation_patch\YT'
#for img_name in os.listdir(img_path):
#    path = os.path.join(img_path, img_name)
#    img = cv2.imread(path)
#    patch_list, all_windows = output_patch(img)
#    
#    test_allwindow.append(all_windows)
#    for i in range(all_windows):
#        cv2.imwrite(os.path.join(save_patch, str(img_num)+'_'+str(i)+'.jpg'),
#                    patch_list[i])
#    img_num += 1

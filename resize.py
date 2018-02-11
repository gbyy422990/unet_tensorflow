#coding: utf-8
#resize image for training.

import os
import cv2
import glob

#folder
img_list  = os.listdir('./data/testlabel1/')


for i in img_list:
    if i[-4:] == '.jpg' and i[:] != '.DS_S.jpg':
        print(i)
        img = cv2.imread('./data/testlabel1/' + i)
        img = cv2.resize(img,(1024,1024))
        

        cv2.imwrite('./data/testlabel/' + str(i[:-4])+'.jpg', img)

        print('Ok')

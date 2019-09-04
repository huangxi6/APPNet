import os
import sys
import numpy as np
import random
import cv2

def generate_seg(label, dataname='person'):
    h, w = label.shape
    seg1 = np.zeros(label.shape)

    for i in range(len(label)):
        for j in range(len(label[0])):
            if label[i][j] > 0:
                seg1[i][j] = 1
                

    return seg1





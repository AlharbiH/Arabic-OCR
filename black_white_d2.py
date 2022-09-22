import cv2
from genericpath import isdir
import os
import shutil

folder = 'd2/'
var = os.listdir(folder)
for v in var:
    
    var2 = os.listdir(folder+v)
    for s in var2:
        img = cv2.imread(folder+v+'/'+s)
        if s == '.DS_Store':
            continue
        bitwiseNot = cv2.bitwise_not(img)
        cv2.imwrite(folder+v+'/'+s, bitwiseNot)
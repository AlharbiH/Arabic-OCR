from fnmatch import fnmatch
import os 
import shutil

folder = 'train'

for f in os.listdir(folder):
    f_name = str(f)
    for i in range(28):
        if 'label_'+str(i)+'.' in f_name:
            shutil.move('train/'+f_name, str(i)+'/'+f_name)

from genericpath import isdir
import os
import shutil

folder = 'd2/'
var = os.listdir(folder)
for v in var:
    if os.path.isdir(folder+v):
        var2 = os.listdir(folder+v)
        for s in var2:
            if s == '.DS_Store':
                    continue
            var3 = os.listdir(folder+v+'/'+s)
            
            for j in var3:
                if j == '.DS_Store':
                    continue
                shutil.move(folder+v+'/'+s+'/'+j, folder+v+'/'+j)




             
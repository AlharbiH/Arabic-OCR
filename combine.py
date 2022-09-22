import os 
import shutil

var1 = os.listdir('d1')
var2 = os.listdir('d2')

for i in var1:
    for j in os.listdir('d1/'+i):
        shutil.copy2('d1/'+ i + '/' + j, 'd3/'+ i + '/' + j)
counter = 1
for i in var2:
    if i == '.DS_Store':
            continue
    for j in os.listdir('d2/'+i):
        if j == '.DS_Store':
            continue
        shutil.copy2('d2/'+ i + '/' + j, 'd3/'+ str(counter) + '/' + j)
    counter += 1
from __future__ import print_function, division

from audioop import bias
import os
import shutil
import random
import torch
import torchvision
from PIL import Image



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import gc



folder = 'd3/test'
try:
    shutil.rmtree('d3/test')
except:
    print("Could Not Delete Test Folder")

   
   

   

os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.manual_seed(0)

   
class DatasetProcess2(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('png')]
            print(f'Found {len(images)} {class_name} examples')
            return images
       
        self.images = {}
        self.class_names = ['pred']
       
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
           
        self.image_dirs = image_dirs
        self.transform = transform
       
   
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])
   
   
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)


train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(255, 255)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(255, 255)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

pred_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(255, 255)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


device = torch.device('cpu')


pred_dirs = {
    'pred': 'd3/pred'
}

pred_dataset = DatasetProcess2(pred_dirs, pred_transform)


dl_pred = torch.utils.data.DataLoader(pred_dataset, batch_size=1, shuffle=True)





def show_preds2():
    model.eval()
    images, labels = next(iter(dl_pred))
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    return preds



model = torch.load('34.pt', map_location="cpu")

index = ['1 alif', '2 ba', '3 ta', '4 tha', '5 gim', '6 ha', '7 kha', '8 dal', '9 thal', '10 ra', '11 zay', '12 sin', '13 shin', '14 sad', '15 dad', '16 da', '17 za', '18 ayn', '19 gayn', '20 fa', '21 qaf', '22 kaf', '23 lam', '24 mim', '25 non', '26 ha', '27 waw', '28 ya']

for folder in os.listdir("d3"):
    file = os.listdir("d3/" + folder)
    try:
        shutil.copy2("d3/" + folder +'/'+file[6], 'd3/pred/1.png')
    except:
        continue
    F_Prediction = str(show_preds2())
    F_Prediction = F_Prediction.replace(', ', '')
    F_Prediction = F_Prediction.replace('tensor([', '')
    F_Prediction = F_Prediction.replace("])", "")
    print("class: "+index[(int(folder)-2)] + " prediction is : " +index[int(F_Prediction)-1])

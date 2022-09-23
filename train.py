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
import os
import gc


folder = 'd3/test'
try:
    shutil.rmtree('d3/test')
except:
    print("Could Not Delete Folder")

   
   

   

os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.manual_seed(0)

print('Using PyTorch version', torch.__version__)
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
root_dir = 'd3'
source_dirs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']

if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
    os.mkdir(os.path.join(root_dir, 'test'))

    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))

    for c in class_names:
        os.mkdir(os.path.join(root_dir, 'test', c))

    for c in class_names:
        images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')]
        selected_images = random.sample(images, 700)
        for image in selected_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.copy2(source_path, target_path)

           
class DatasetProcess(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('png')]
            print(f'Found {len(images)} {class_name} examples')
            return images
       
        self.images = {}
        self.class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
       
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



train_dirs = {


    '1': 'd3/1',
    '2': 'd3/2',
    '3': 'd3/3',
    '4': 'd3/4',
    '5': 'd3/5',
    '6': 'd3/6',
    '7': 'd3/7',
    '8': 'd3/8',
    '9': 'd3/9',
    '10': 'd3/10',
    '11': 'd3/11',
    '12': 'd3/12',
    '13': 'd3/13',
    '14': 'd3/14',
    '15': 'd3/15',
    '16': 'd3/16',
    '17': 'd3/17',
    '18': 'd3/18',
    '19': 'd3/19',
    '20': 'd3/20',
    '21': 'd3/21',
    '22': 'd3/22',
    '23': 'd3/23',
    '24': 'd3/24',
    '25': 'd3/25',
    '26': 'd3/26',
    '27': 'd3/27',
    '28': 'd3/28'
    
}



train_dataset = DatasetProcess(train_dirs, train_transform)

test_dirs = {
    '1': 'd3/test/1',
    '2': 'd3/test/2',
    '3': 'd3/test/3',
    '4': 'd3/test/4',
    '5': 'd3/test/5',
    '6': 'd3/test/6',
    '7': 'd3/test/7',
    '8': 'd3/test/8',
    '9': 'd3/test/9',
    '10': 'd3/test/10',
    '11': 'd3/test/11',
    '12': 'd3/test/12',
    '13': 'd3/test/13',
    '14': 'd3/test/14',
    '15': 'd3/test/15',
    '16': 'd3/test/16',
    '17': 'd3/test/17',
    '18': 'd3/test/18',
    '19': 'd3/test/19',
    '20': 'd3/test/20',
    '21': 'd3/test/21',
    '22': 'd3/test/22',
    '23': 'd3/test/23',
    '24': 'd3/test/24',
    '25': 'd3/test/25',
    '26': 'd3/test/26',
    '27': 'd3/test/27',
    '28': 'd3/test/28'


}

device = torch.device('cpu')

test_dataset = DatasetProcess(test_dirs, test_transform)


pred_dirs = {
    'pred': 'd3/pred'
}

pred_dataset = DatasetProcess2(pred_dirs, pred_transform)


batch_size = 10

dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
dl_pred = torch.utils.data.DataLoader(pred_dataset, batch_size=12, shuffle=True)
print('Number of training batches', len(dl_train))
print('Number of test batches', len(dl_test))


class_names = train_dataset.class_names



images, labels = next(iter(dl_train))


images, labels = next(iter(dl_test))


model = torchvision.models.mobilenet_v2(pretrained=True)


model.fc = torch.nn.Linear(in_features=512, out_features=28)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)


def show_preds():
    model.eval()
    images, labels = next(iter(dl_test))
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)
    print(preds)
   
def show_preds2():
    model.eval()
    images, labels = next(iter(dl_pred))
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    return preds



def train(epochs):
    ste = []
    acc = []
    ls = []
    tr_acc = []
    tr_ls = []
    print('Starting training..')
    for e in range(0, epochs):
        print('='*20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)

        train_loss = 0.
        val_loss = 0.
        train_accuracy = 0

        model.train().to(device)
        

        for train_step, (images, labels) in enumerate(dl_test):
            model.eval()
            
            optimizer.zero_grad()
            outputs= model(images.to(device))
            loss = loss_fn(outputs, labels.to(device))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_accuracy += sum((preds.to(device) == labels.to(device)).cpu().numpy())
            
            
            if train_step % 20 == 0:
                print('Evaluating at step', train_step)

                accuracy = 0

                model.eval()

                for val_step, (images, labels) in enumerate(dl_test):
                    

                    outputs = model(images.to(device))
                    loss = loss_fn(outputs, labels.to(device))
                  
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    accuracy += sum((preds.to(device) == labels.to(device)).cpu().numpy())

                val_loss /= (val_step + 1)
                accuracy = accuracy/len(test_dataset)
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

                show_preds()

                model.train().to(device)
                ste.append(train_step)
                acc.append(accuracy*100)
                ls.append(val_loss)
                if accuracy >= 0.998 and val_loss <= 0.01:
                    print('Performance condition satisfied, stopping..')
                    print("Saving the model")
                    torch.save(model, 'plot18.pt')
                    

                    break
        print(acc)
        print(ls)
            
        train_loss /= (train_step + 1)

        print(f'Training Loss: {train_loss:.4f}')
    print('Training complete..')




train(epochs=1)


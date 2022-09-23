# Arabic OCR

Arabic Handwritten Character Recognition using Deep Learning

## Hardware Requirements 

RAM: 8GB or Higher - 32GB Recommended.

CPU: Intel core i5 or Higher - Intel core i7 Recommended.

GPU: Nvidia GTX 1660 SUPER or Higher - RTX 3060 Recommended.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirements.txt
```

## Usage

* Note: the first four steps are optional if the final dataset [d3](google.com) is installed 

1- Install HACD Dataset and extract it 

2- Install HIJJA2 Dataset and extract it

3- Run the following commands to prepare dataset for combination

```bash
python move_to_folders_d1.py

python move_to_folders_d1.py
```
4- Combine the datasets using the following command

```bash
python combine.py
```
5- Start Model training

```bash
python train.py
```

6- Start Model testing 

```bash 
python test.py 
```

Notes:

1- To generate different types of model change line 255 in train.py into one of the following 
```python 
model = torchvision.models.mobilenet_v2(pretrained=True) # MobileNet
model = torchvision.models.resnet18(pretrained=True) # Resnet18
model = torchvision.models.resnet34(pretrained=True) # Resnet34
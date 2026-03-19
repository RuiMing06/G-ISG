from __future__ import print_function
import numpy as np
import time
import torch
import torch.nn as nn
import random
import torch
import os
import iDLG.reconstruction as DLG
import IG
import ISG
import ISG_NAS
from ISG_NAS.generator import trainer, BNlayers_mean_var
from ISG.reconstruction_algorithms import _label_to_onehot
from utils.nb_utils import save_images,calculate_iqa
from utils import config
from arch.model import convert_relu_to_sigmoid
from torchvision import datasets

from arch.model import VBMLP,VBResNet18,VBResNet34,BLConvNet4,BLConvNet8
import torchvision.transforms as transforms
from ISG_NAS.cross_skip import skip
import numpy as np
from PIL import Image

device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
setup = dict(device=device, dtype=torch.float)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

batch_num = 1
data_name = 'NWPU'
cf = config.config_NAS

# Define transformations
mean=[0.368, 0.381, 0.344] 
std=[0.145, 0.136, 0.132] 
dm, ds = torch.as_tensor(mean, **setup)[:, None, None], torch.as_tensor(std, **setup)[:, None, None]
transform = transforms.Compose([
    # transforms.Resize((576,576)), # for AID
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=std)
]) 

if data_name == 'NWPU':
    image_size = (3, 256, 256)
    class_num = 45
    image_num = 700
    # dataset = datasets.ImageFolder(root='../datasets/NWPU-RESISC45', transform=transform)
    imgs = [Image.open('result/NWPU/origin_{}/{}.jpg'.format(batch_num,i)) for i in range(batch_num)]
    image_batch = torch.stack([transform(imgs[i]) for i in range(batch_num)]).to(device)
elif data_name == 'AID':
    image_size = (3, 576, 576)
    class_num = 30
    image_num = 500
    # dataset = datasets.ImageFolder(root='../datasets/AID',transform=transform)
    imgs = [Image.open('result/AID/origin_{}/{}.jpg'.format(batch_num,i)) for i in range(batch_num)]
    image_batch = torch.stack([transform(imgs[i]) for i in range(batch_num)]).to(device)
elif data_name == 'RSI':
    image_size = (3, 128, 128)
    class_num = 45
    image_num = 1000
    # dataset = datasets.ImageFolder(root='../datasets/RSI-CB128',transform=transform)
    imgs = [Image.open('result/RSI/origin_{}/{}.jpg'.format(batch_num,i)) for i in range(batch_num)]
    image_batch = torch.stack([transform(imgs[i]) for i in range(batch_num)]).to(device)
elif data_name == 'imagenet':
    image_size = (3, 256, 256)
    class_num = 1000
    image_num = 100
    saved_data = torch.load("../datasets/imagenet-data/data_sample100_resolution256.pt")
    saved_label = torch.load("../datasets/imagenet-data/label_sample100_resolution256.pt")
    # image_batch = torch.stack([saved_data[i] for i in range(batch_num)]).to(device)
    # label_batch = torch.tensor([saved_label[i] for i in range(batch_num)])
elif data_name == 'UC':
    image_size = (3, 256, 256)
    class_num = 21
    image_num = 100
    # dataset = datasets.ImageFolder(root='../datasets/UCMerced_LandUse/Images',transform=transform)
    imgs = [Image.open('result/UC/origin_{}/{}.jpg'.format(batch_num,i)) for i in range(batch_num)]
    image_batch = torch.stack([transform(imgs[i]) for i in range(batch_num)]).to(device)
else:
    image_size = (3, 64, 64)
    class_num = 10
    image_num = 10
    imgs = [Image.open('test.bmp')]
    image_batch = torch.stack([transform(imgs[i]) for i in range(batch_num)]).to(device)

# images = {}
# for i in range(class_num):
#     images[i] = []
# for idx,label in enumerate(dataset.targets):
#     images[label].append(dataset[idx][0]) 

labels = random.sample(range(class_num),batch_num) # 不同类
# labels = [random.choice(range(class_num)) for i in range(batch_num)] # 随机类
# print(labels)

# for RSI-CB128 64
# labels = [40, 7, 1, 17, 15, 14, 8, 6, 43, 34, 5, 37, 27, 2, 1, 5, 13, 14, 32, 38, 1, 35, 12, 41, 44, 34, 26, 14, 28, 37, 17, 0, 10, 44, 27, 21, 17, 9, 13, 21, 6, 5, 24, 6, 22, 22, 38, 16, 2, 29, 34, 7, 24, 5, 35, 18, 40, 39, 23, 36, 12, 4, 2, 42]
# for NWPU-RESISC45 4 8 16
labels = [i for i in range(batch_num)]
# labels = [15, 9, 18, 6, 14, 2, 7, 8]

# image_batch = torch.stack([images[i][random.choice(range(image_num))] for i in labels]).to(device) # different classes
# image_batch = torch.stack([images[l][k + i] for i in range(batch_num)]).to(device) # same classes

label_batch = torch.tensor(labels) # different classes
# label_batch = torch.tensor([l] * batch_num) # same classes

label_batch = _label_to_onehot(label_batch, class_num).to(device)

# from torchvision.models import resnet18,resnet34
# victim_model = resnet18(num_classes=class_num)
# victim_model = VBResNet18(num_classes=class_num)
# victim_model = BLConvNet4(num_classes=class_num)
victim_model = VBResNet34(num_classes=class_num)
# victim_model = VBMLP(num_classes=class_num,data_shape=image_size)
convert_relu_to_sigmoid(victim_model)
victim_model.to(device)

parameters = dict(multiplier=[], models=[], eps=[])
for name, p in victim_model.named_parameters(): 
    if 'multiplier' in name:
        parameters['multiplier'].append(p)
    # learnable intermediate noise
    elif 'eps' in name or 'epsb' in name:
    # hook the intermediate noise which is going to be optimized
        parameters['eps'].append(p)
    # the model parameters
    else:
        parameters['models'].append(p)

# victim_model.train()
victim_model.zero_grad()
# output,_,_ = victim_model(image_batch)
output = victim_model(image_batch)
target_loss = loss_fn(output, label_batch) + victim_model.loss()
# target_loss = loss_fn(output, label_batch)
input_gradient = torch.autograd.grad(target_loss, parameters['models'])

dict = victim_model.state_dict()
means_vars = BNlayers_mean_var(dict)
# means_vars = []

if cf['method'] in ['DLG','iDLG']:
    victim_model.flagVB = 1
    rec_machine = DLG.GradientReconstructor(victim_model, (dm, ds), cf, num_images=batch_num)
    dummy_images, dummy_labels = rec_machine.reconstruct(input_gradient, label_batch, img_shape=image_size)

elif cf['method'] == 'IG':
    victim_model.flagVB = 1
    rec_machine = IG.GradientReconstructor(victim_model, (dm, ds), cf, num_images=batch_num)
    dummy_images = rec_machine.reconstruct(input_gradient, label_batch, img_shape=image_size)

elif cf['method'] == 'ISG':
    victim_model.flagVB = 2
    rec_machine = ISG.GradientReconstructor(victim_model, parameters, (dm, ds), cf, num_images=batch_num)
    dummy_images, dummy_labels = rec_machine.reconstruct(input_gradient, label_batch, img_shape=image_size)

elif cf['method'] == 'GINAS':
    victim_model.flagVB = 2
    # rec_machine = ISG_NAS.GradientReconstructor(victim_model, parameters, (dm, ds), cf, num_images=batch_num)
    # dummy_images, dummy_labels = rec_machine.reconstruct(input_gradient, label_batch, img_shape=image_size)
    
    my_trainer = trainer(cf,victim_model,parameters,batch_num)
    dummy_images = my_trainer.attack_training(input_gradient, label_batch, image_size, means_vars, device, image_batch)

# save_images(image_batch,'origin_{}'.format(batch_num),data_name)

save_images(dummy_images,cf['method']+'_{}_vb34'.format(batch_num),data_name)

image_batch.clamp_(min=0,max=1)
dummy_images.clamp_(min=0,max=1)
score = calculate_iqa(image_batch, dummy_images)
print(score)








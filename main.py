from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader 
from BaseUNet import BaseUNet
import torch
from torchvision import transforms


model = BaseUNet(3,3)
model.cuda()

print("Number of parameters in the model:",sum(p.numel() for p in model.parameters()))

data = CIFAR10('./data',download=True,transform=transforms.ToTensor())
dataset= DataLoader(data,batch_size=3, shuffle=True)
for img,_ in dataset:
    img= img.cuda()
    output=model(img)
    
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader 
from DownBlock import DownBlock
from TransitionBlock import TransitionBlock
import torch
from torchvision import transforms


model = TransitionBlock(3,8)
model.cuda()


print("Number of parameters in the model:",sum(p.numel() for p in model.parameters()))

data = CIFAR10('./data',download=True,transform=transforms.ToTensor())
dataset= DataLoader(data,batch_size=128, shuffle=True)
for img,_ in dataset:
    img= img.cuda()
    output=model(img)
    
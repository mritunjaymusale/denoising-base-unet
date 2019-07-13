from torchvision.datasets import CIFAR10
from BaseUNet import BaseUNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from math import log10
import utils
import torch 
from torch.autograd import Variable


utils.makeDirectories()
batch_size = 128

model = BaseUNet(3, 3)
model.cuda()
model.eval()
model.load_state_dict(torch.load( './saved_model/cifar10_base_unet.pth'))

data = CIFAR10('./data', download=True, transform=utils.to_32_32_transform(),train=False)
dataset = DataLoader(data, batch_size=batch_size, shuffle=True)

avg_psnr = 0
for img, _ in dataset:
    img = img.cuda()
    noise_img = Variable(
        img+img.data.new(img.size()).normal_(0.0, 0.1)).cuda()
    output = model(noise_img)
    MSE_loss = nn.MSELoss()(output, img)
    psnr = 10 * log10(1 / MSE_loss.data)
    avg_psnr += psnr

print('MSE_loss:{:.4f}'.format(MSE_loss.data))
print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(dataset)))
ground_truth,noise,unet_output=utils._to_img(img,noise_img,output)
utils._save_image(ground_truth,noise,unet_output,epoch='Testing')


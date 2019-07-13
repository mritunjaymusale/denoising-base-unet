from torchvision.datasets import CIFAR10
from torch.autograd import Variable
from torch.utils.data import DataLoader
from BaseUNet import BaseUNet
import torch
from torchvision import transforms
from torch import nn
from math import log10
import utils


utils.makeDirectories()
num_epochs = 50
batch_size = 128
learning_rate = 1e-3

model = BaseUNet(3, 3)
model.cuda()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)

print("Number of parameters in the model:", sum(p.numel()
                                                for p in model.parameters()))

data = CIFAR10('./data', download=True, transform=transforms.ToTensor())
dataset = DataLoader(data, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    avg_psnr = 0
    for img, _ in dataset:
        img = img.cuda()
        noise_img = Variable(
            img+img.data.new(img.size()).normal_(0.025, 0.1081)).cuda()
        output = model(noise_img)

        loss = criterion(output, img)
        MSE_loss = nn.MSELoss()(output, img)
        psnr = 10 * log10(1 / MSE_loss.data)
        avg_psnr += psnr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data, MSE_loss.data))
    print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(dataset)))
    ground_truth,noise,unet_output=utils._to_img(img,noise_img,output)
    utils._save_image(ground_truth,noise,unet_output,epoch)
    torch.save(model.state_dict(), './saved_model/cifar10_base_unet.pth')
        # add other noise removing methods for comparison 
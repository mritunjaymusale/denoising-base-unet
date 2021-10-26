from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from BaseUNet import BaseUNet
import torch
from torch import nn
import utils
import piq

torch.manual_seed(0000)
utils.makeDirectories()
num_epochs = 200
batch_size = 256
learning_rate = 1e-3

model = BaseUNet(3, 3)
model.cuda()
MSE_loss = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)

print("Number of parameters in the model:", sum(p.numel()
                                                for p in model.parameters()))

data = CIFAR10('./data', download=True,
               transform=utils.to_32_32_transform(), train=True)
dataset = DataLoader(data, batch_size=batch_size,
                     shuffle=True, num_workers=2, pin_memory=True)

std = 1.
mean = 0.
noise_tensor = torch.randn([ 3, 32, 32]) * std + mean
for epoch in range(num_epochs):
    avg_psnr = 0
    for img, _ in dataset:
        img = img.cuda()

        # generate noisy image
        noise = torch.empty_like(img)
        noise.normal_(0, 0.1)
        noise_img = img +noise

        # model datafeed
        output = model(noise_img)

        mse_loss = MSE_loss(output, img)

        # PSNR

        psnr = piq.psnr(output, img,
                        data_range=255, reduction='none')
        print("PSNR :", psnr.mean().item())
        print('epoch [{}/{}], mse_loss:{:.4f}'
              .format(epoch + 1, num_epochs, mse_loss.item()))

        # update gradients
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

    ground_truth, noise, unet_output = utils._to_img(img, noise_img, output)
    utils._save_image(ground_truth, noise, unet_output, epoch)
    torch.save(model.state_dict(), './saved_model/cifar10_base_unet.pth')
    

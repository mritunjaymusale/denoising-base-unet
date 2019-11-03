from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from BaseUNet import BaseUNet
import torch
from torch import nn
import utils

torch.manual_seed(0000)
utils.makeDirectories()
num_epochs = 200
batch_size = 128
learning_rate = 1e-3

model = BaseUNet(3, 3)
model.cuda()
MSE_loss = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)

print("Number of parameters in the model:", sum(p.numel()
                                                for p in model.parameters()))

data = CIFAR10('./data', download=True, transform=utils.to_32_32_transform(),train=True)
dataset = DataLoader(data, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)


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
       
        mse_loss =MSE_loss(output, img)
        
        # PSNR
        psnr = 10 * torch.log10(1 / mse_loss.data)
        avg_psnr += psnr
        
        # update gradients
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
    print('epoch [{}/{}], mse_loss:{:.4f}'
            .format(epoch + 1, num_epochs, mse_loss.data))
    print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(dataset)))
    ground_truth,noise,unet_output=utils._to_img(img,noise_img,output)
    utils._save_image(ground_truth,noise,unet_output,epoch)
    torch.save(model.state_dict(), './saved_model/cifar10_base_unet.pth')
        # add other noise removing methods for comparison 
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from BaseUNet import BaseUNet
import utils
from torchvision.datasets import CIFAR10
import piq

pl.utilities.seed.seed_everything(seed=0000, workers=True)

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # TODO: fix channels later
        self.model = BaseUNet(3,3)

    def forward(self, x):
        noise = torch.empty_like(x)
        noise.normal_(0, 0.1)
        noise_img = x +noise
        # model datafeed
        output = self.model(noise_img)
        return output
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img = train_batch[0]
        
        # generate noisy image
        noise = torch.empty_like(img)
        noise.normal_(0, 0.1)
        noise_img = img +noise

        # model datafeed
        output = self.model(noise_img)
        # PSNR

        psnr = piq.psnr(output, img,
                        data_range=255, reduction='none')
        
        loss = F.mse_loss(output, img)
        self.log('PSNR', psnr.mean().item())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        img = val_batch[0]
        
        # generate noisy image
        noise = torch.empty_like(img)
        noise.normal_(0, 0.1)
        noise_img = img +noise
        
        # model datafeed
        output = self.model(noise_img)
        
        # PSNR

        psnr = piq.psnr(output, img,
                        data_range=255, reduction='none')
        
        loss = F.mse_loss(output,img)
        self.log('PSNR', psnr.mean().item())
        self.log('val_loss', loss)
        return loss
        

batch_size = 256
# data
train_data = CIFAR10('./data', download=True,
               transform=utils.to_32_32_transform(), train=True)
train_loader = DataLoader(train_data, batch_size=batch_size,
                     shuffle=True, num_workers=2, pin_memory=False)

val_data = CIFAR10('./data', download=True,
               transform=utils.to_32_32_transform(), train=False)
val_loader = DataLoader(val_data, batch_size=batch_size,
                     shuffle=True, num_workers=2, pin_memory=False)



# model
model = MyModel()

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=1.0)
trainer.fit(model, train_loader, val_loader)
        

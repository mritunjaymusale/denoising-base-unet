import os
from torchvision.utils import save_image

def to_img(x):
    x = x.view(x.size(0), 3, 32, 32)
    return x

def makeDirectories():
    if not os.path.exists('./result_images'):
        os.mkdir('./result_images')

def _save_image(ground_truth,noise,unet_output,epoch):
    save_image(ground_truth, './result_images/ground_truth_{}.png'.format(epoch))
    save_image(noise, './result_images/noise_{}.png'.format(epoch))
    save_image(unet_output, './result_images/unet_output_{}.png'.format(epoch))

def _to_img(img,noise_img,output):
	ground_truth = to_img(img.cpu().data)
	noise = to_img(noise_img.cpu().data)
	unet_output = to_img(output.cpu().data)
	return ground_truth,noise,unet_output
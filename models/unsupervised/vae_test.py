# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:05:26 2020

@author: wangxin
"""
#%% 导入包
from AutoEncoder import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt

#%%
img_transform = transforms.Compose([
    transforms.ToTensor()])
path = "../../dataset"
batch_size = 128
dataset = MNIST(path, transform=img_transform, train = True, download=False)
dataIter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
imgs = dataset.data[:6].numpy()
labels = dataset.targets[:6].numpy()

_, axes = plt.subplots(2, 3)
for i in range(2):
    for j in range(3):
        axes[i][j].imshow(imgs[i*3 + j], cmap='gray')
        axes[i][j].set_title("True: " + str(labels[i*3+j]))
        axes[i][j].get_xaxis().set_visible(False)
        axes[i][j].get_yaxis().set_visible(False)
plt.show()
#%%
encoder_structure = [784, 512, 64]
decoder_structure = [20, 64, 512, 784]
model = VAE(encoder_structure, decoder_structure, 20)
# opt = torch.optim.SGD(model.parameters(), 0.01)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)
model.train(model, dataIter, nn.MSELoss(size_average=False), opt, 50)
#%%
shape = (6, 20)
z_mean = torch.rand(shape, device='cuda')

rand_z = torch.randn(shape,device='cuda') + z_mean
# rand_z = torch.randn(shape,device='cuda')
gen_x = model.decode(rand_z).cpu()
rand_img = to_image(gen_x).detach().numpy()
# rand_img = (rand_img * 255 / (rand_img.max() - rand_img.min())).astype(np.uint8)
_ ,axes = plt.subplots(2, 3)
for i in range(2):
    for j in range(3):
        axes[i][j].imshow(rand_img[i*3+j], cmap='gray')
plt.show()
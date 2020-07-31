#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch import nn

class VAE(nn.Module):
    # 使用全链接网络
    def __init__(self, encoder_structure, decoder_structure, hidden_num):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(1, len(encoder_structure)):
            self.encoder.add_module("linear"+str(i), nn.Linear(encoder_structure[i-1], encoder_structure[i]))
            self.encoder.add_module("relu"+str(i), nn.ReLU())

        self.z_layer = nn.Linear(encoder_structure[-1], hidden_num)
        self.log_var_layer = nn.Linear(encoder_structure[-1], hidden_num)

        self.decoder = nn.Sequential()
        for i in range(1, len(decoder_structure)):
            self.decoder.add_module("linear"+str(i), nn.Linear(decoder_structure[i-1], decoder_structure[i]))
            if(i < len(decoder_structure)-1): self.decoder.add_module("relu"+str(i), nn.ReLU())
    
    def forward(self, x):
        self.z_mean, self.z_log_var = self.encode(x)
        z = self._reparameters(self.z_mean, self.z_log_var)
        self.x_mean = self.decode(z)
        return self.z_mean, self.z_log_var, z, self.x_mean

    def encode(self, x):
        code = self.encoder(x)
        z_mean = self.z_layer(code)
        z_log_var = self.log_var_layer(code)
        return z_mean, z_log_var

    def decode(self, z):
        x_mean = self.decoder(z)
        return x_mean

    def loss(self, x, recon_func):
        KL_loss = -0.5 * torch.sum(1 + self.z_log_var - self.z_mean.pow(2) - self.z_log_var.exp())
        recon_loss = recon_func(self.x_mean, x)
        return KL_loss + recon_loss

    def _reparameters(self, z_mean, z_log_var):
        z0 = torch.randn_like(z_mean)
        return z_mean + z0 * torch.exp(0.5*z_log_var)
    
    def train(self, net, dataIter, recon_loss, optimizer, epoches):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("training on %s" %(device))
        net = net.to(device)
        train_loss = [0.]*epoches
        for epoch in range(epoches):
            cnt = 0
            for batch_idx, (data, label) in enumerate(dataIter):
                # 前向
                data = data.view(data.size(0), -1).to(device)
                z_mean, z_log_var, z, x_mean = net(data)
                loss = net.loss(data, recon_loss)
                # 反向
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss[epoch] += loss.cpu().item()
                if((batch_idx+1) % 100 == 0):
                    print("epoch : {0} | #batch : {1} | batch average loss: {2}"
                          .format(epoch, batch_idx, loss.cpu().item()/len(data)))
            # train_loss[epoch] /= len(dataIter.dataset)
            print("Epoch : {0} | epoch average loss : {1}"
                  .format(epoch, train_loss[epoch] / len(dataIter.dataset)))
def to_image(x):
    return x.view(x.shape[0], 28, 28)



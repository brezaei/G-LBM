from __future__ import (
    absolute_import,
    division,
    print_function
)
import numpy as np 
import os
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from tqdm import *
from dataset import *

__all__ = ['loss_fn', 'Trainer']




def ConvOutSize(in_size, ConvLayNum, kernel, stride, padding):
    """
    Parameters
    in_size: input height and width given as a tuple (height, width)
    -----
    """
    height = in_size[0]
    width =  in_size[1]
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    for _ in range(ConvLayNum):
        height = (height - kernel[0] + 2*padding)/stride + 1
        width = (width - kernel[1] + 2*padding)/stride + 1
    return (height, width)


def loss_fn(original_seq, recon_seq, z_mean, z_logvar, z, beta = 1.0):
    """
    Loss function consists of 3 parts, the reconstruction term that is the l1-loss between the generated background model 
    and the original images
    the KL divergence of z
    Nuclear norm of the z
    Loss = {l_1 + KL of z + sum(SVD of z)} / batch_size
    Prior of z is a spherical zero mean unit variance Gaussian
    """
    batch_size = original_seq.size(0)
    l1 = F.l1_loss(recon_seq, original_seq, reduction='sum')
    kld = -0.5* beta * torch.sum(1 + z_logvar - torch.pow(z_mean, 2) -torch.exp(z_logvar))
    nuclear = torch.mm(z, torch.transpose(z, 0, 1))
    nuclear = torch.trace(nuclear)
    return (l1 + kld + nuclear)/batch_size, kld/batch_size, l1/batch_size, nuclear/batch_size
    
class Trainer(object):
    def __init__(self, model, train, test, train_loader, test_loader, im_size, check_freq  = 5, 
    load_checkpoint_path=None, epochs = 200,batch_size = 8, learning_rate= 0.001, 
    recon_path='./recon/', checkpoints='./checkpoints/', device=torch.device('cuda:0')):
        self.model = model
        self.train = train
        self.test = test
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.recon_path = recon_path
        self.device = device
        self.checkpoints = checkpoints
        self.load_checkpoint_path = load_checkpoint_path
        self.start_epoch = 0
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.epoch_losses = []

    def save_checkpoint(self, epoch):
        file_name = '{}model_epoch_{}.pth'.format(self.checkpoints, epoch+1)
        if isinstance(model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        torch.save({
            'epoch':epoch+1,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'losses': self.epoch_losses},
            file_name
        )
    
    def load_ckeckpoint(self):
        try:
            print("Loading checkpoint from '{}".format(self.load_checkpoints_path))
            ckeckpoint = torch.load(self.load_checkpoint_path)
            self.start_epoch = ckeckpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming training from epoch {}".format(self.start_epoch))
        except:
            print("No checkpoint exists at {}. Start fresh training".format(self.load_checkpoint_path))
            self.start_epoch =  0
    def recon_frame(self, epoch, original):
        with torch.no_grad():
            recon = self.model(original)
            image = torch.cat((original, recon), dim=0)
            image = image.view(2,3, im_size[0], im_size[1])
            os.makedirs(os.path.dirname('{}/epoch{}.png'.format(self.recon_path, epoch)), exist_ok=True)
            torchvision.utils.save_image(image, '{}/epoch{}.png'.format(self.recon_path, epoch))
    def train_model(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            kld = []
            nuclear = []
            print("Running Epoch : {}".format(epoch +1))
            for _, data in tqdm(enumerate(self.trainloader, 1)):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_x, z_mean, z_logvar, z = self.model(data)
                loss, kld, l1, nuclear = loss_fn(data, recon_x, z_mean, z_logvar, z)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                kld.append(kld.item())
                l1.append(l1.item())
                nuclear.append(nuclear.item())
            meanloss = np.mean(losses)
            meankld = np.mean(kld)
            l1mean = np.mean(l1)
            meannuclear = np.mean(nuclear)
            self.epoch_losses.append(meanloss)
            print("Epoch {} :: Average Loss:{}, KL of z:{}, L1 Norm:{}, Nuclear norm:{}"
            .format(epoch+1, meanloss, meankld, l1mean, meannuclear))
            if (epoch+1)%5 == 0:
                self.save_checkpoint(epoch)
                self.model.eval()
                if self.test is not None:
                    sample = self.test[int(torch.randint(0,len(self.test),(1,)).item())]
                    sample = torch.unsqueeze(sample, 0)
                    sample = sample.to(self.device)
                    self.recon_frame(epoch+1, sample)
                self.model.train()
        print("Training is complete")


class CDW(data.Dataset):
    """
    Dataset class for loading a video clip
    """
    def __init__(self, path, listOfFolders, n_frames=120, transform = None):
        """
        path: path to the data folder to be loaded
        size: number of the images to be loaded
        transform: 
        """
        self.path = path
        self.length = size
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = torch.load(self.path+'')


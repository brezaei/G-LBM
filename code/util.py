from __future__ import (
    absolute_import,
    division,
    print_function
)
import numpy as np 
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
import math
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
        height = math.floor((height - kernel[0] + 2*padding)/stride + 1)
        width = math.floor((width - kernel[1] + 2*padding)/stride + 1)
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


class SBMnet_singleVideo(Dataset):
    """
    Dataset class for loading a video clip
    """
    def __init__(self, root_dir, n_frames, transform = None):
        """
        path: path to the data folder to be loaded
        n_frames: number of frames of the video 
        transform: 
        """
        self.root_dir = root_dir
        self.length = n_frames
        self.img_dir = os.path.join(self.root_dir, 'input')
        self.id_list = range(0, self.length)
    
    def get_image(self, idx):
        im_file = os.path.join(self.img_dir, 'in%06d.jpg' %idx)
        assert os.path.exist(im_file)
        return io.imread(im_file)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.get_image(self.id_list[idx])

class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):        
        indices = []
        for i in range(len(end_idx)-1):
            start = end_idx[i]
            end = end_idx[i+1] - seq_length
            indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices
        
    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.indices)


class MyDataset(Dataset):
    def __init__(self, image_paths, seq_length, transform, length):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        x = torch.stack(images)
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        
        return x, y
    
    def __len__(self):
        return self.length


root_dir = './video_data_test/'
class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]

class_image_paths = []
end_idx = []
for c, class_path in enumerate(class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
            # Add class idx to paths
            paths = [(p, c) for p in paths]
            class_image_paths.extend(paths)
            end_idx.extend([len(paths)])

end_idx = [0, *end_idx]
end_idx = torch.cumsum(torch.tensor(end_idx), 0)
seq_length = 10

sampler = MySampler(end_idx, seq_length)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = MyDataset(
    image_paths=class_image_paths,
    seq_length=seq_length,
    transform=transform,
    length=len(sampler))

loader = DataLoader(
    dataset,
    batch_size=1,
    sampler=sampler
)

for data, target in loader:
    print(data.shape)
from __future__ import (
    absolute_import,
    division,
    print_function
)
import numpy as np 
import os, sys
import glob
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
import math
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.optim.lr_scheduler as lr_sched
import tqdm 

__all__ = ['loss_fn_singleVideo', 'Trainer', 'ConvOutSize', 'dataset_singleVideo']

DEBUG=False

def ConvOutSize(in_size, ConvLayNum, kernel, stride, padding):
    """
    Parameters
    in_size: (int, int) = input height and width given as a tuple (height, width)
    ConvLayNum: int =  number of the convolutional layers
    kernel: (int, int) = kernel size of the conv layers
    stride : int = stride 
    padding: int = padding
    -----
    """
    height = in_size[0]
    width =  in_size[1]
    pad_list=[[0,0] for _ in range(ConvLayNum)]
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    for i in range(ConvLayNum):
        height = (height - kernel[0] + 2*padding)/stride + 1
        width = (width - kernel[1] + 2*padding)/stride + 1
        #print("size after {} conv is computed as:{},{}".format(i,height, width))
        if height - math.floor(height) != 0.0:
            pad_list[i][0] = 1
        if width - math.floor(width) != 0.0:
            pad_list[i][1] = 1 
        height = math.floor(height)
        width = math.floor(width)
    return (height, width), pad_list


def loss_fn_singleVideo(original_seq, recon_seq, z_mean, z_logvar, z, beta = 1.0, alpha=1.0, eps = 1e-2):
    """
    Prameters:
    origninal_seq [B, 1, 3, H, W] = input video frmes
    recon_seq [B, 1, 3, H, W] = generated background
    z_mean, z_logvar [B, 1, z_dim] = mean and logvar of the 
    Loss function consists of 3 parts, the reconstruction term that is the l1-loss between the generated background model 
    and the original images
    the KL divergence of z
    Nuclear norm of the z 
    Loss = {alpha * l_1 + beta * KL of z + sum(SVD of z)} / batch_size
    Prior of z is a spherical zero mean unit variance Gaussian
    """
    #///TODO: adding the l21-norm(recon_seq(t)-recon_seq(t-1))
    #print("original_seg:{}, reconstructed_seq:{}, z:{}".format(original_seq.shape, recon_seq.shape, z.shape)) 
    batch_size = original_seq.shape[0]
    # sqeeze all the values to [B,3, H, W]
    original_seq = original_seq.transpose(0, 1).squeeze()
    recon_seq = recon_seq.transpose(0, 1).squeeze()
    z_mean = z_mean.transpose(0, 1).squeeze()
    z_logvar = z_logvar.transpose(0, 1).squeeze()
    z = z.transpose(0, 1).squeeze()
    #print(torch.mean(recon_seq[0, 0]), torch.mean(original_seq))
    l1 = F.l1_loss(recon_seq, original_seq, reduction='sum')
    kld = -0.5* beta * torch.sum(1 + z_logvar - torch.pow(z_mean, 2) -torch.exp(z_logvar))
    #nuclear = torch.mm(z, torch.transpose(z, 0, 1))
    #nuclear = torch.trace(nuclear)
    _, s, _ = torch.svd(z, some=True, compute_uv=True)
    nuclear = alpha * torch.sum(s)
    rank = torch.sum(s > eps)
    #rank = torch.tensor(1)
    return (l1+kld+nuclear)/batch_size, kld/batch_size, l1/batch_size, nuclear/batch_size, rank
    
def loss_fn_multipleVideo(original_seq, recon_seq, z_mean, z_logvar, z, beta = 1.0, alpha=1.0):
    """
    Prameters:
    origninal_seq [B, f, 3, H, W] = input video frmes
    recon_seq [B, f, 3, H, W] = generated background
    z_mean, z_logvar [B, f, z_dim] = mean and logvar of the 
    Loss function consists of 3 parts, the reconstruction term that is the l1-loss between the generated background model 
    and the original images
    the KL divergence of z
    Nuclear norm of the z 
    Loss = {alpha * l_1 + beta * KL of z + sum(SVD of z)} / batch_size
    Prior of z is a spherical zero mean unit variance Gaussian
    """
    ImportError("not yet implemented")

def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn

def load_checkpoint(model=None, optimizer=None, filename='checkpoint'):
    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        print("==> Done")
    else:
        raise FileNotFoundError

    return it, epoch


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class Trainer(object):
    def __init__(self, model, optimizer, im_size, loss_fn, lr_scheduler = None, 
                 lr_warmup_scheduler=None, bnm_scheduler=None, warmup_epoch=-1, grad_norm_clip=1.0, check_freq=10,
                 learning_rate=0.001, recon_path='../recon/', ckpt_dir='../checkpoints/'):
        self.model = model
        self.im_size = im_size
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_scheduler = lr_warmup_scheduler
        self.bnm_scheduler = bnm_scheduler
        self.warmup_epoch = warmup_epoch
        self.learning_rate = learning_rate
        self.recon_path = recon_path
        self.ckpt_dir = ckpt_dir
        self.check_freq =  check_freq
        self.start_epoch = 0
        self.start_it = 0
        self.optimizer = optimizer
        self.grad_norm_clip = grad_norm_clip
        self.epoch_losses = []

    def save_checkpoint(self, epoch, it):
        file_name = os.path.join(self.ckpt_dir, 'checkpoint_epoch_{:d}.pth'.format(epoch+1))
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        torch.save({
            'epoch':epoch+1,
            'it': it+1,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'losses': self.epoch_losses},
            file_name
        )
    
    def load_ckeckpoint(self, load_checkpoint_path=None):
        try:
            print("Loading checkpoint from '{}".format(load_checkpoint_path))
            ckeckpoint = torch.load(load_checkpoint_path)
            self.start_epoch = ckeckpoint['epoch']
            self.start_it = checkpoint['it']
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming training from epoch {}".format(self.start_epoch))
        except:
            print("No checkpoint exists at {}. Start fresh training".format(self.load_checkpoint_path))
            self.start_epoch =  0
            self.start_it = 0
    def recon_frame(self, epoch, sample_id, original):
        with torch.no_grad():
            recon,_, _, _ = self.model(original)
            image = torch.cat((original, recon), dim=0)
            image = image.view(2,3, self.im_size[0], self.im_size[1])
            os.makedirs(os.path.dirname('{}/epoch{}_sample{}.png'.format(self.recon_path, epoch, sample_id)), exist_ok=True)
            torchvision.utils.save_image(image, '{}/epoch{}_sample{}.png'.format(self.recon_path, epoch, sample_id))
    def train_model(self, n_epochs, train_loader, test_loader=None, alpha=1.0):
        it = self.start_it
        self.model.train()
        with tqdm.trange(self.start_epoch, n_epochs, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:
            for epoch in tbar:
                losses = []
                klds = []
                nuclears = []
                l1s = []
                ranks = []
                print("Running Epoch : {}".format(epoch +1))
                for _, data in enumerate(train_loader, start=0):
                    if DEBUG:
                        print("\n\n\n---------------- Running Iteration : {}".format(it))
                        print("WEIGHTS WEIGHTS WEIGHTS WEIGHTS WEIGHTS WEIGHTS ")
                        # print conv weights
                        print("\nconv:")
                        for convLayer in self.model.conv:
                            l = convLayer.model[0].weight
                            print("min:{}, max:{}".format(torch.max(l).item(), torch.min(l).item()))
                        
                        # print conv_fc weights
                        print("\nconv_fc:")
                        l = self.model.conv_fc.model[0].weight
                        print("min:{}, max:{}".format(torch.max(l).item(), torch.min(l).item()))

                        # print deconv_fc weitghs
                        print("\ndeconv_fc:")
                        for fcLayer in self.model.deconv_fc:
                            l = fcLayer.model[0].weight
                            print("min:{}, max:{}".format(torch.max(l).item(), torch.min(l).item()))

                        # print deconv weitghs
                        print("\ndeconv:")
                        for deconvLayer in self.model.deconv:
                            l = deconvLayer.model[0].weight
                            print("min:{}, max:{}".format(torch.max(l).item(), torch.min(l).item()))

                    data = data.cuda(non_blocking=True).float()
                    if self.lr_warmup_scheduler is not None and epoch < self.warmup_epoch:
                        self.lr_warmup_scheduler.step(it)
                        cur_lr = self.lr_warmup_scheduler.get_lr()[0]
                    elif self.lr_scheduler is not None:
                        cur_lr = self.lr_scheduler.get_lr()[0]
                    else:
                        cur_lr = None
                    # self.model.zero_grad()
                    self.optimizer.zero_grad()
                    recon_x, z_mean, z_logvar, z = self.model(data)
                    # print("z_mean:{}, z_var:{}, z:{}".format(torch.mean(z_mean.squeeze(0), axis=0),
                    #  torch.mean(z_logvar.squeeze(0), axis=0),torch.mean(z.squeeze(0), axis=0)))
                    loss, kld, l1, nuclear, rank = self.loss_fn(data, recon_x, z_mean, z_logvar, z, alpha=alpha)
                    loss.backward()
                    if DEBUG:
                        print("\nGRADS GRADS GRADS GRADS GRADS GRADS GRADS GRADS \n")
                        if isinstance(self.model.parameters(), torch.Tensor):
                            parameters = [self.model.parameters()]
                        else:
                            parameters = self.model.parameters()
                            inf_error = False
                            for p in parameters:
                                print("before clip size:{}, mean value:{}, standard deviation:{}"
                                .format(p.grad.data.shape, p.grad.data.mean(), p.grad.data.std()))
                                if torch.isinf(p.grad.data.std()):
                                    inf_error = True
                            if inf_error:
                                print("\nINF GRAD DETECTED!!\n")
                                # sys.exit(1)
                    clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                    if DEBUG:
                        print("\n")
                        if isinstance(self.model.parameters(), torch.Tensor):
                            parameters = [self.model.parameters()]
                        else:
                            parameters = self.model.parameters()
                        for p in parameters:
                            print("after clip size:{}, mean value:{}, standard deviation:{}"
                            .format(p.grad.data.shape, p.grad.data.mean(), p.grad.data.std()))
                        print("\n\n\n")
                    self.optimizer.step()
                    it += 1

                    losses.append(loss.item())
                    klds.append(kld.item())
                    l1s.append(l1.item())
                    nuclears.append(nuclear.item())
                    ranks.append(rank.item())
                    if DEBUG:
                        print("LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS ")
                        print(dict(loss = loss.item(), kld=kld.item(), l1=l1.item(), nuclear=nuclear.item()))
                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.set_postfix(dict(loss = loss.item(), kld=kld.item(), l1=l1.item(), nuclear=nuclear.item()))
                    tbar.refresh()
                if self.lr_scheduler is not None and self.warmup_epoch <= epoch:
                    self.lr_scheduler.step(epoch)

                if self.bnm_scheduler is not None:
                    self.bnm_scheduler.step(it)
                    print('bn_momentum', self.bnm_scheduler.lmbd(epoch), it)
                meanloss = np.mean(losses)
                meankld = np.mean(klds)
                meanl1 = np.mean(l1s)
                meannuclear = np.mean(nuclears)
                meanrank = np.mean(ranks)
                self.epoch_losses.append(meanloss)
                print("current lr:{}, Epoch: {},  Average Loss:{}, KL of z:{}, L1 Norm:{}, Nuclear norm:{}, rank:{}"
                .format(cur_lr, epoch+1, meanloss, meankld, meanl1, meannuclear, meanrank))
                if (epoch+1)% self.check_freq == 0:
                    self.save_checkpoint(epoch, it)
                    self.model.eval()
                    if test_loader is not None:
                        for i, sample in enumerate(test_loader):
                            sample = sample.cuda(non_blocking=True).float()
                            #sample = torch.unsqueeze(sample, 0)
                            self.recon_frame(epoch+1, i, sample)
                    self.model.train()
        print("Training is complete")


class dataset_singleVideo(Dataset):
    """
    Dataset class for loading frames of a video 
    """
    def __init__(self, path, img_format = 'jpg', transform = None):
        """
        path(string): path to the video frame folder to be loaded
        transform ( callable, optional): optional transformation to be applied on a sample
        """
        self.path = path
        self.transform = transform
        self.img_list = sorted(glob.glob(os.path.join(self.path, '*.{}'.format(img_format))))
    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.path, self.img_list[idx])
        frame = io.imread(img_name)
        frame = frame.transpose((2, 0, 1))
        frame = frame/255.0
        frame = torch.from_numpy(frame).unsqueeze(0)
        if self.transform:
            frame = self.transform(frame)
        return frame

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


'''
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
'''
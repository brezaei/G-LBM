import multiprocessing
multiprocessing.set_start_method('spawn', True)

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import os
import numpy as np
import argparse
from skimage import io
# from torchviz import make_dot
import torchvision
import matplotlib.pyplot as plt

from model_deep import LR_VAE
import util 

from torchviz import make_dot
from graphviz import Digraph

###############################################################
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("-batch_size", type=int, default=120, help="Batch size [default: 120]")
parser.add_argument("-epochs", type=int, default=200, required=True, help="Number of epochs to train for [default=200]")
parser.add_argument("-alpha", type=float, default=1.0, help="rank regularizer [default: 1.0]")
parser.add_argument("-beta", type=float, default=0.8, help="KL divergence regularizer [default: 0.8]")
parser.add_argument("-vid_path", type=str, required=True, help="video frame folder path to be processed")
parser.add_argument("-ckpt", type=str, default=None, help="continue training from this checkpoint")
parser.add_argument("-recon_path", type=str, default='../recon/', help="path to the folder for reconstructed frames")
parser.add_argument("-ckpt_dir", type=str, default='../checkpoints/', help="path to the folder for saving the checkpoints")
parser.add_argument('--mgpus', action='store_true', help='whether to use multiple gpus')
parser.add_argument('--train_with_eval', action='store_true', help='whether to train with evaluation')
parser.add_argument("-weight_decay",type=float,default=0.001,help="L2 regularization coeff [default: 0.0]")
parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate [default: 1e-2]")
parser.add_argument("-lr_decay",type=float,default=0.5,help="Learning rate decay gamma [default: 0.5]")
parser.add_argument("-decay_step_list", nargs='+', type=int,default=[50, 100, 150, 200, 250, 300],help="Learning rate decay step [default: 50, 100, 150, 200, 250, 300]")
parser.add_argument("-clip", type=float, default=1.0, help="clip value for the gradient [default=1.0]")
parser.add_argument("-im_format", type=str, default='jpg', help="image format of the video frames")
args = parser.parse_args()
###############################################################


# Parameters
params = {
        'optimizer': 'adam',
        'shuffle': False,
        'check_freq':30,
        'num_workers': 4,
        'num_smpls':10,
        'lr_warmup':False,
        'lr_clip':1e-6,
        'warmup_min':0.0002,
        'warmup_epoch':5,
        'bn_decay':0.5,
        'bn_momentum':0.1,
        'bnm_clip':0.01,
        'bn_decay_step_list':[50, 100, 150, 200, 250, 300],
        'momentum':0.9
}
np.random.seed(seed=0)
def create_dataloader():

    # create dataloader
    train_set = util.dataset_singleVideo(path=args.vid_path, img_format=args.im_format, transform=None)
    if len(train_set) > 11000:
        train_set = [train_set[idx] for idx in range(11000)]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
                              num_workers=params['num_workers'], shuffle=params['shuffle'], drop_last=True)

    print("++++++++++++ Training Data is loaded containing {} batches of size:{}".format(len(train_loader), args.batch_size))
    #print(args.train_with_eval)
    if args.train_with_eval:
        test_set = util.dataset_singleVideo(path=args.vid_path, img_format=args.im_format, transform=None)
        indx = list(np.random.randint(0, len(test_set), params['num_smpls']))
        test_set = [test_set[idx] for idx in indx]
        print('frame numbers:{} are sampled for the validation phase'.format(indx))
        #print(test_set[0])
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True,
                                 num_workers=params['num_workers'])
    else:
        test_loader = None
    return train_loader, test_loader

def create_optimizer(model):

    if params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif params['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                              momentum=params['momentum'])
    else:
        raise NotImplementedError

    return optimizer

def create_scheduler(optimizer, total_steps, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in args.decay_step_list:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * args.lr_decay
        return max(cur_decay, params['lr_clip'] / args.lr)

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in params['bn_decay_step_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * params['bn_decay']
        return max(params['bn_momentum'] * cur_decay, params['bnm_clip'])
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    bnm_scheduler = util.BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return lr_scheduler, bnm_scheduler

###########################################################################
if __name__ == "__main__":
    # torch.manual_seed(2020)
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    print('CUDA_VISIBLE_DEVICES=%s' % gpu_list)    
# find the size of the input frames
    im = io.imread(os.path.join(args.vid_path, 'in000001.{}'.format(args.im_format)))
    in_size = im.shape
    in_size = in_size[0:2]
    print("input size is:{}".format(in_size))
# create dataloader & network & optimizer
    train_loader, test_loader = create_dataloader()
    model = LR_VAE(
                    kernel=4, 
                    stride =2, 
                    z_dim=120, 
                    in_size= in_size,
                    frames = 1, 
                    nonlinearity=None
    )
    optimizer = create_optimizer(model)

    if args.mgpus:
        model = nn.DataParallel(model)
    model.cuda()

    # x = torch.zeros(120, 3, 240, 320, dtype=torch.float, requires_grad=False, device='cuda')
    # y=model(x)
    # g=make_dot(y)
    # g.view()

    start_epoch = it = 0
    last_epoch = -1
    lr_scheduler, bnm_scheduler = create_scheduler(optimizer, total_steps=len(train_loader) * args.epochs,
                                                   last_epoch=last_epoch)
    if args.ckpt is not None:
        pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        it, start_epoch = util.load_checkpoint(pure_model, optimizer, filename=args.ckpt)
        last_epoch = start_epoch + 1

    if params['lr_warmup']:
        lr_warmup_scheduler = util.CosineWarmupLR(optimizer, T_max=params['warmup_epoch'] * len(train_loader),
                                eta_min=params['warmup_min'])
    else:
        lr_warmup_scheduler = None

    # start training

    print('************************ start training *******************************')

    trainer = util.Trainer(
        model, 
        optimizer,  
        in_size, 
        util.loss_fn_singleVideo,  
        lr_scheduler = lr_scheduler,
        lr_warmup_scheduler=lr_warmup_scheduler, 
        bnm_scheduler=None, 
        warmup_epoch=-1, 
        grad_norm_clip=args.clip, 
        check_freq  = params['check_freq'],
        learning_rate= args.lr, 
        recon_path= args.recon_path, 
        ckpt_dir= args.ckpt_dir
        )

    if args.ckpt is not None:
        trainer.load_ckeckpoint(load_checkpoint_path=args.ckpt)

    trainer.train_model(args.epochs, train_loader, test_loader = test_loader, alpha=args.alpha)
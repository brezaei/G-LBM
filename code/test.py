import multiprocessing
multiprocessing.set_start_method('spawn', True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from skimage import io
import torchvision

from model_deep import LR_VAE
import util 


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("-vid_path", type=str, required=True, help="video frame folder path to be processed")
parser.add_argument("-ckpt", type=str, default=None, help="model checkpoint to be used for test")
parser.add_argument("-result_path", type=str, default='../result/', help="path to the folder for generated background frames")
parser.add_argument("-im_format", type=str, default='jpg', help="image format of the video frames")
parser.add_argument('--mgpus', action='store_true', help='whether to use multiple gpus')
parser.add_argument("-device", type=str, default='gpu', help="device for mapping the model and loading data: 'gpu' or 'cpu' [default:'gpu']")
args = parser.parse_args()

# create dataloader

if __name__ == "__main__":

    data_set = util.dataset_singleVideo(path=args.vid_path, img_format=args.im_format, transform=None)
    data_loader = DataLoader(data_set, batch_size=1, pin_memory=True,num_workers=1,
                            shuffle=False, drop_last=False)

    # find the size of the video frames

    im = io.imread(os.path.join(args.vid_path, 'in000001.{}'.format(args.im_format)))
    in_size = im.shape
    in_size = in_size[0:2]

    model = LR_VAE(
                        kernel=4, 
                        stride =2, 
                        z_dim=120, 
                        in_size= in_size,
                        frames = 1, 
                        nonlinearity=None,
                        sampling_z=True
    )
    #print("========= model is loaded!")
    if args.mgpus:
        model = nn.DataParallel(model)

    if args.device == 'cpu':
        model.cpu()
    else:
        model.gpu()

    pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    it, start_epoch = util.load_checkpoint(pure_model, filename=args.ckpt, device=args.device)

    model.eval()
    for i, data in enumerate(data_loader):
        if args.device=='cpu':
            data = data.float()
        else:
            data = data.cuda(non_blocking=True).float()
        
        bg, _, _, _ = model(data)
        bg_im = bg.view(3, in_size[0], in_size[1])
        os.makedirs(os.path.dirname('{}/bg{:06d}.png'.format(args.result_path,i)), exist_ok=True)
        torchvision.utils.save_image(bg_im, '{}/bg{:06d}.png'.format(args.result_path,i))


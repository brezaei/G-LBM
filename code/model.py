"""
Written by Behnaz Rezaei (brezaei@ece.neu.edu)
Unsupervised Deep Probabilistic Modeling of the background in videos recorded by static camera
developed based on the concept of Variational autoencoders
"""
from __future__ import (
    absolute_import,
    division,
    print_function
)
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function, Variable
import sys

from util import *

from torch.nn import Parameter
import torch.utils.data
from torch import nn, optim

# parameters of the encoder and decoder
h_layer_1 = 32
h_layer_2 = 32
h_layer_3 = 64
h_layer_4 = 64
h_layer_5 = 2400
"""
    defining the encoder and decoder blocks
"""
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x
# A block consisting of convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class ConvUnit(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel, 
        stride= 1, 
        padding=0, 
        batchnorm: bool = False,
        bias: bool = False, 
        nonlinearity=nn.LeakyReLU(0.2)
        ):
        super(ConvUnit, self).__init__()
        bias = bias and (not batchnorm)
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=bias),
                    nn.BatchNorm2d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=bias), nonlinearity)
    def forward(self, x):
        return self.model(x)

# A block consisting of a transposed convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class ConvUnitTranspose(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel, 
        stride=1, 
        padding=0, 
        out_padding=0, 
        batchnorm:bool=False,
        bias:bool=True, 
        nonlinearity=nn.LeakyReLU(0.2)
        ):
        super(ConvUnitTranspose, self).__init__()
        bias = bias and (not batchnorm)
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding, bias=bias),
                    nn.BatchNorm2d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding, bias=bias), nonlinearity)
    def forward(self, x):
        return self.model(x)

# A block consisting of an affine layer, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class LinearUnit(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        batchnorm:bool=False, 
        nonlinearity=nn.LeakyReLU(0.2)
        ):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features, bias= False),
                    nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features, bias=True), nonlinearity)

    def forward(self, x):
        return self.model(x)


# defining the VAE model
class LR_VAE(nn.Module):
    r"""Network Architecture:
        PRIOR OF Z:
            The prior of z is a Gaussian with mean 0 and variance I
        CONVOLUTIONAL ENCODER FOR CONDITIONAL DISTRIBUTION q(z|x):
            The convolutional encoder consists of 4 convolutional layers with 256 layers and a kernel size of 4 
            Each convolution is followed by a batch normalization layer and a LeakyReLU(0.2) nonlinearity. 
            For the 3,64,64 frames (all image dimensions are in channel, width, height) in the sprites dataset the following dimension changes take place
            
            3,320,240 ->  ->  ->  ->  (where each -> consists of a convolution, batch normalization followed by LeakyReLU(0.2))

            The 8,8,256 tensor is unrolled into a vector of size 8*8*256 which is then made to undergo the following tansformations
            
             ->  ->  (where each -> consists of an affine transformation, batch normalization followed by LeakyReLU(0.2))

        CONVOLUTIONAL DECODER FOR CONDITIONAL DISTRIBUTION p(x| z)
            The architecture is symmetric to that of the convolutional encoder. The vector f is concatenated to each z_t, which then undergoes two subsequent
            affine transforms, causing the following change in dimensions
            
             ->  ->  (where each -> consists of an affine transformation, batch normalization followed by LeakyReLU(0.2))

            The 8*8*256 tensor is reshaped into a tensor of shape 256,8,8 and then undergoes the following dimension changes 

             ->  ->  ->  -> 3,320,240 (where each -> consists of a transposed convolution, batch normalization followed by LeakyReLU(0.2)
            with the exception of the last layer that does not have batchnorm and uses tanh nonlinearity)

    Hyperparameters:
        z_dim: Dimension of the background encoding of a frame z. z has the shape (batch_size, z_dim)   
        nonlinearity: Nonlinearity used in convolutional and deconvolutional layers, defaults to LeakyReLU(0.2)
        in_size: (Height and width) of each frame in the video
        h_layer_*: Number of channels in the convolutional and deconvolutional layers
        conv_dim: The convolutional encoder converts each frame into an intermediate encoding vector of size conv_dim, i.e,
                  The initial video tensor (batch_size, num_channels, in_size[0], in_size[1]) is converted to (batch_size, conv_dim)

    Optimization:
        The model is trained with the Adam optimizer with a learning rate of 0.0001, betas of 0.9 and 0.999, with a batch size of 120 for 200 epochs

    """
    def __init__(self, kernel=4, stride =2, z_dim=120, in_size=(320,240),
                 frames = 1, nonlinearity=None):
        super(LR_VAE, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.z_dim = z_dim
        self.in_size = in_size
        self.frames = frames
        self.nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        # ///TODO: Check if only one affine transform is sufficient.
        self.z_mean = LinearUnit(h_layer_5, self.z_dim, batchnorm=False)
        self.z_logvar = LinearUnit(h_layer_5, self.z_dim, batchnorm=False)
        # ///TODO: check if it is better to add padding in here instead of output padding in deconv
        self.conv = nn.Sequential(
                ConvUnit(3, h_layer_1, self.kernel, self.stride), 
                ConvUnit(h_layer_1, h_layer_2, self.kernel, self.stride, nonlinearity=self.nl), 
                ConvUnit(h_layer_2, h_layer_3, self.kernel, self.stride, nonlinearity=self.nl),
                ConvUnit(h_layer_3, h_layer_4, self.kernel, self.stride, nonlinearity=self.nl), 
                )
        self.final_conv_size, self.pad_list = ConvOutSize(in_size, 4 , self.kernel, self.stride, 0)

        self.conv_fc = LinearUnit(h_layer_4 * (self.final_conv_size[0]*self.final_conv_size[1]), h_layer_5)

        self.deconv_fc = nn.Sequential(LinearUnit(self.z_dim, h_layer_5, batchnorm=False),
                LinearUnit(h_layer_5, h_layer_4 * (self.final_conv_size[0]*self.final_conv_size[1]), False))
        #///TODO: try the nn.Tanh() as the nonlinearity, it scales the output in [-1, 1] compared to sigmoid which is [0, 1]
        self.deconv = nn.Sequential(
                ConvUnitTranspose(h_layer_4, h_layer_3, self.kernel, self.stride, padding=0, out_padding=self.pad_list[3]),
                ConvUnitTranspose(h_layer_3, h_layer_2, self.kernel, self.stride, padding=0, out_padding=self.pad_list[2]),
                ConvUnitTranspose(h_layer_2, h_layer_1, self.kernel, self.stride, padding=0, out_padding=self.pad_list[1]),
                ConvUnitTranspose(h_layer_1, 3, self.kernel, self.stride, padding=0, out_padding=self.pad_list[0], nonlinearity=nn.Sigmoid()))
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def encoder(self, x):
        # The frames are unrolled into the batch dimension for batch processing such that x goes from
        # [batch_size, frames, channels, height, width] to [batch_size * frames, channels, height, width]
        x = x.view(-1, 3, self.in_size[0], self.in_size[1])
        x = self.conv(x)
        x = x.view(-1, h_layer_4 * (self.final_conv_size[0]*self.final_conv_size[1]))
        x = self.conv_fc(x)
        mean = self.z_mean(x)
        logvar = self.z_logvar(x)
        # The frame dimension is reintroduced and x shape becomes [batch_size, frames, z_dim]
        # This technique is repeated at several points in the code
        mean = mean.view(-1, self.frames, self.z_dim)
        logvar = logvar.view(-1, self.frames, self.z_dim)
        return mean, logvar
    
    def decoder(self, z):
        # The frames are unrolled into the batch dimension for batch processing such that z goes from
        # [batch_size, frames, z_dim] to [batch_size * frames, z_dim]
        x = z.view(-1, self.z_dim)
        x = self.deconv_fc(z)
        x = x.view(-1, h_layer_4, self.final_conv_size[0], self.final_conv_size[1])
        x = self.deconv(x)
        return x.view(-1, self.frames, 3, self.in_size[0], self.in_size[1])

    def reparam(self, z_mean, z_logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(z_logvar)
            std = torch.exp(0.5*z_logvar)
            z = z_mean + eps*std
            return z
        else:
            return z_mean
    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparam(z_mean, z_logvar)
        return self.decoder(z), z_mean, z_logvar, z





        



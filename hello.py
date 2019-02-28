#=========================================================
# import libs
#=========================================================
import argparse
import os
import numpy as np
import math
import itertools
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from coco_text_dataset import coco_text_dataset

#=========================================================
# init program
#=========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--bbox_count', type=int, default=1, help='bbox count')
parser.add_argument('--text_dim', type=int, default=100, help='text embeding dim')
parser.add_argument('--style_dim', type=int, default=256, help='style dim')
parser.add_argument('--img_size', type=int, default=256, help='image size')
parser.add_argument('--load', type=bool, default=False, help='load from saved checkpoint')
opt = parser.parse_args()
print("parsed option:", opt)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

#=========================================================
# Util Functions
#=========================================================
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def calc_code_dim(opt):
    return (5 + opt.text_dim) * opt.bbox_count + opt.style_dim

def getTimeStamp():
    timestemp = time.strftime(R"%m-%d_%H-%M-%S", time.localtime())
    return timestemp

#=========================================================
# Util Class
#=========================================================

class Flat(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        size = x.size()
        if len(size) == 4:
            x = x.view(-1, size[1] * size[2] * size[3])
        return x

#=========================================================
# ResBlock Class
#=========================================================
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        if out_channel % 4 != 0:
            raise Exception('ResBlock output should divide by 4')
        
        bottle = int(out_channel/4)
        self.relu = nn.LeakyReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, bottle, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottle, 0.8),
            self.relu,

            nn.Conv2d(bottle, bottle, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottle),
            self.relu,

            nn.Conv2d(bottle, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottle),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False, stride=stride),
                nn.BatchNorm2d(out_channel, 0.8),
            )
    
    def forward(self, x):
        x = self.net(x)
        x += self.shortcut(x)
        x = self.relu(x)
        return x

class ResBlockBackward(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        if out_channel % 4 != 0:
            raise Exception('ResBlock output should divide by 4')

        bottle = int(out_channel/4)
        self.relu = nn.LeakyReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv2d(out_channel, bottle, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottle, 0.8),
            self.relu,

            nn.ConvTranspose2d(bottle, bottle, kernel_size=3, bias=False, padding=1, stride=stride, output_padding=1),
            nn.BatchNorm2d(bottle, 0.8),
            self.relu,
            
            nn.Conv2d(bottle, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel, 0.8),
        )

        self.shortcut = nn.Sequential()
        if(in_channel != out_channel or stride != 1):
            self.shortcut = nn.Sequential(
                #nn.ConvTranspose2d(out_channel, in_channel, kernel_size=1, stride=stride, bias=False, output_padding=1),
                nn.Conv2d(out_channel, in_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channel, 0.8),
                nn.Upsample(scale_factor=stride),
            )
    
    def forward(self, x):
        x = self.net(x)
        x += self.shortcut(x)
        x = self.relu(x)
        return x

#=========================================================
# input: code[bbox(con), text(con), style(latent)] Nx((5+text_dim)*bbox_count+style_dim)x14x14
# output: image
#=========================================================
class GenX(nn.Module):
    def __init__(self):
        super(GenX, self).__init__()

        def resBlockBackward(repeat, in_channel, out_channel, stride=2):
            layers = []
            for _ in range(repeat-1):
                layers.append(ResBlockBackward(in_channel=out_channel, out_channel=out_channel, stride=1))
            layers.append(ResBlockBackward(in_channel=in_channel, out_channel=out_channel, stride=stride))
            return nn.Sequential(*layers)
        
        self.code_dim = calc_code_dim(opt)

        self.net = nn.Sequential(
            nn.BatchNorm2d(self.code_dim),
            #l1
            nn.Conv2d(self.code_dim, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024, 0.8),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024, 0.8),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024, 0.8),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(512, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048, 0.8),
            nn.LeakyReLU(inplace=True),

            #upsample 14 > 28
            #resblock cfg[3]=8
            resBlockBackward(8, 1024, 2048),
            
            #upsample 28 > 56
            #resblock cfg[2]=23
            resBlockBackward(23, 512, 1024),
            
            #upsample 56 > 112
            #resblock cfg[1]=4
            resBlockBackward(4, 256, 512),
            
            #resblock cfg[0]=3
            resBlockBackward(3, 64, 256, stride=1),            
            
            #upsample 112 > 448
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3),
            nn.Tanh(),
        )

    def forward(self, bbox, text, style):
        x = torch.cat((bbox, text, style), 1)
        x = self.net(x)
        return x

#=========================================================
# input: image
# output: code[bbox(con), text(con), style(latent)]
#=========================================================
class GenZ(nn.Module):
    def __init__(self):
        super(GenZ, self).__init__()

        def resBlock(repeat, in_channel, out_channel, stride=2):
            layers = []
            layers.append(ResBlock(in_channel=in_channel, out_channel=out_channel, stride=stride))
            for _ in range(repeat-1):
                layers.append(ResBlock(in_channel=out_channel, out_channel=out_channel, stride=1))
            return nn.Sequential(*layers)

        self.code_dim = calc_code_dim(opt)    

        self.net = nn.Sequential(
            nn.BatchNorm2d(3),
            #downsample 112 < 448
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(64),
            
            #resblock cfg[0]=3
            resBlock(3, 64, 256, stride=1),
            
            #downsample 56 < 112
            #resblock cfg[1]=4
            resBlock(4, 256, 512),
            
            #downsample 28 < 56
            #resblock cfg[2]=23
            resBlock(23, 512, 1024),

            #downsample 14 < 28
            #resblock cfg[3]=8
            resBlock(8, 1024, 2048),
            
            #l1
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 1024, kernel_size=3, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(1024, 0.8),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 1024, kernel_size=3, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(1024, 0.8),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(512, 1024, kernel_size=3, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(1024, 0.8),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(1024, self.code_dim, kernel_size=1, bias=False),
            # nn.BatchNorm2d(self.code_dim, 0.8),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.net(x)
        bbox = self.x[:,0:5*opt.bbox_count,:,:]
        text = self.x[:,5*opt.bbox_count:5*opt.bbox_count+opt.text_dim*opt.bbox_count,:,:]
        if opt.bbox_count > 1:
            raise Exception('not impl')
        style = self.x[:,(5+opt.text_dim)*opt.bbox_count:(5+opt.text_dim)*opt.bbox_count+opt.style_dim,:,:]
        return bbox, text, style

#=========================================================
# input: image
# output: valid(1), bbox(5,con), text(100,con)
#=========================================================
class DiscX(nn.Module):
    def __init__(self):
        super(DiscX, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return nn.Sequential(*block)

        self.net_front = nn.Sequential(
            discriminator_block(3, 16, bn=False),
            discriminator_block(16, 32),
            discriminator_block(32, 64),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**5

        # Output layers
        self.net_valid = nn.Sequential(
            Flat(),
            nn.Linear(256*ds_size**2, 1)
        )
        self.net_bbox = nn.Sequential(
            nn.Conv2d(256, opt.bbox_count * 5, kernel_size=1)
        )
        self.net_text = nn.Sequential(
            nn.Conv2d(256, opt.text_dim * opt.bbox_count, kernel_size=1)
        )

    def forward(self, x):
        x = self.net_front(x)
        valid = self.net_valid(x)
        bbox = self.net_bbox(x)
        text = self.net_text(x)
        return valid, bbox, text

#=========================================================
# input: bbox, text, code
# output: valid(1)
#=========================================================
class DiscZ(nn.Module):
    def __init__(self):
        super(DiscZ, self).__init__()

        dim = calc_code_dim(opt)
        ds_size = opt.img_size // 2**5

        self.net = nn.Sequential(
            nn.Conv2d(dim, 256, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 128, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            Flat(),
            
            nn.Linear(128*ds_size**2, 1)
        )
    
    def forward(self, bbox, text, style):
        x=torch.cat((bbox, text, style), 1)
        return self.net(x)

#**note** Disc Opt Rule: MSE to 0(fake) MSE to 1(real)

#=========================================================
# main program
#=========================================================
if __name__ == "__main__":
    #=====================================================
    # init model
    #=====================================================

    if opt.load:
        pass
    else:
        genX = GenX()
        genZ = GenZ()
        discX = DiscX()
        discZ = DiscZ()
        for i in [genX, genZ, discX, discZ]:
            i.apply(weights_init_normal)
            i.train()
    
    #=====================================================
    # load data
    #=====================================================
    
    data = coco_text_dataset()

    #=====================================================
    # init train
    #=====================================================

    #optimizers
    optCycle = torch.optim.Adam(itertools.chain(genX.parameters(), genZ.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    optDz = torch.optim.Adam(discZ.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optDx = torch.optim.Adam(discX.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optInfo = torch.optim.Adam(itertools.chain(genX.parameters(), discX.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    optBbox = torch.optim.Adam(genZ.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    #losses
    lossGAN = torch.nn.MSELoss()
    lossCycle = torch.nn.L1Loss()
    lossIdentity = torch.nn.L1Loss()
    lossInfoCon = nn.MSELoss()
    lossInfoCat = nn.CrossEntropyLoss()
    #lambdas
    lambda_cat = 1
    lambda_con = 0.1
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
    #move to cuda
    if cuda:
        for i in [lossCycle, lossInfoCat, lossInfoCon, genX, genZ, discX, discZ]:
            i.cuda()
        
    raise Exception("WIP")

    #=====================================================
    # train loop
    #=====================================================
    while True:
        #=================================================
        # Batch: Read Data
        #=================================================

        batch = data.batch(opt.batch_size)
        batch_image = batch['img']
        batch_bbox = batch['bbox']
        batch_text = batch['text']
        batch_text = batch['style']

        #=================================================
        # Cycle: Train Generator
        # https://github.com/aitorzip/PyTorch-CycleGAN/
        #=================================================
        
        optCycle.zero_grad()

        #==== Identity loss
        same_B = netG_A2B(batch_image)
        loss_identity_B = lossIdentity(same_B, real_B)*5.0
        same_A = netG_B2A(real_A)
        loss_identity_A = lossIdentity(same_A, real_A)*5.0

        #==== GAN loss

        #==== Cycle loss

        #==== Total loss

        optCycle.step()

        #=================================================
        # Cycle: train Disc Z
        #=================================================

        #=================================================
        # Cycle: train Disc X
        #=================================================

        #=================================================
        # Info: train Info
        #=================================================

        #=================================================
        # YOLO: train BBox
        #=================================================

        #=================================================
        # Logging
        #=================================================        

        pass
#!/usr/bin/env python
# coding: utf-8

# # Latium
# Latium Project 2019. Written by Heejun Lee
# # Compile
# ## Import Libs

# In[1]:


import sys
import argparse
import os
import numpy as np
import math
import itertools
import time
import random
import gc
import util
import datetime

import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models.vgg as vgg

import display
import TFBoard
import _pickle as cPickle

from coco_text_dataset import coco_text_dataset

def in_notebook():
    return 'ipykernel' in sys.modules

if in_notebook():
    get_ipython().run_line_magic('matplotlib', 'inline')


# ## Init Program

# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--batch_update', type=int, default=8, help='size of the batches')
parser.add_argument('--gan_update', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--lr_update', default=False, action='store_true', help='lupdate learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--bbox_count', type=int, default=1, help='bbox count')
parser.add_argument('--text_dim', type=int, default=196, help='text embeding dim')
parser.add_argument('--style_dim', type=int, default=32, help='style dim')
parser.add_argument('--latent_dim', type=int, default=512, help='latent dim')
parser.add_argument('--img_size', type=int, default=448, help='image size')
parser.add_argument('--vgg_depth', type=int, default=10, help='image size')
parser.add_argument('--save_interval', type=int, default=300, help='image size')
parser.add_argument('--pic_interval', type=int, default=120, help='image size')
parser.add_argument('--load', default=False, action='store_true', help='load from saved checkpoint')
parser.add_argument('--load_data', default=False, action='store_true', help='load data from saved cache')
parser.add_argument('--no_save', default=False, action='store_true', help='load data from saved cache')
parser.add_argument('--cfg_genX', default='3,4,23,8', help='delimited list input', type=str)
parser.add_argument('--cfg_genZ', default='3,4,23,8', help='delimited list input', type=str)

if in_notebook():
    print("parsing is passed. running on notbook")
    parser.add_argument(
        'strings',
        metavar='STRING',
        nargs='*',
        help='String for searching',
    )
    parser.add_argument(
            '-f',
            '--file',
            help='Path for input file. First line should contain number of lines to search in'
        )
    opt = parser.parse_args()
else:
    opt = parser.parse_args()
opt.cfg_genX = [int(item) for item in opt.cfg_genX.split(',')]
opt.cfg_genZ = [int(item) for item in opt.cfg_genZ.split(',')]
if len(opt.cfg_genX) != 4 or len(opt.cfg_genZ) != 4:
    raise Exception('cfg must be len 4')
print("parsed option:", opt)

opt.cuda = True if torch.cuda.is_available() else False
if opt.cuda:
    device_cuda = torch.device('cuda:0')
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor


# ## Util Functions

# In[3]:


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def calc_code_dim(opt):
    return (5 + opt.text_dim + opt.style_dim) * opt.bbox_count + opt.latent_dim

def getTimeStamp():
    timestemp = time.strftime(R"%m-%d_%H-%M-%S", time.localtime())
    return timestemp


# ## Util Class

# In[4]:


class Flat(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        size = x.size()
        if len(size) == 4:
            x = x.view(-1, size[1] * size[2] * size[3])
        return x
is_debug = True

class Print(nn.Module):
    def __init__(self, msg='message'):
        super().__init__()
        self.msg = msg
    def forward(self, x):
        global is_debug
        if is_debug:
            print("Module Debug!", self.msg, "size:", x.size())
        return x

class Logger():
    def __init__(self, logdir='./temp', outputLimit=32):
        self.logdir = logdir
        self.stack = 0
        self.outputLimit = outputLimit
        self.startTime = datetime.datetime.now()
        self.tfboard = None
    
    def print(self, msg):
        if self.stack > self.outputLimit:
            display.clear()
            self.stack = 0
        self.stack += 1
        timeMsg = str(datetime.datetime.now()-self.startTime)
        print("[{}] {}".format(timeMsg, msg))
    
    def log_image(self, dic, global_step=None):
        if self.tfboard is None:
            self.tfboard = TFBoard.Tensorboard(self.logdir)
            self.print('tfboard inited: ' + self.logdir)
        for k in dic:
            self.tfboard.log_image(k, dic[k], global_step)
        
    def log(self, scalarDict, global_step=None):
        if self.tfboard is None:
            self.tfboard = TFBoard.Tensorboard(self.logdir)
            self.print('tfboard inited: ' + self.logdir)
        output = ""
        for i, k in enumerate(sorted(scalarDict.keys())):
            v = scalarDict[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if global_step is not None:
                self.tfboard.log_scalar(k, v, global_step, flush=False)
            t = k+" : "+str(v)
            if i!=0:
                t+=', \n'
            output = t+output
        if global_step is not None:
            output = "(" + str(global_step) + " steps) " + output
        if self.tfboard is not None:
            self.tfboard.flush()
        self.print(output)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []
    
    def to(self, device):
        for d in self.data:
            d.to(device)
    
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class VGG16Feature(nn.Module):
    def __init__(self, depth=10000):
        super().__init__()
        self.vgg = vgg.vgg16(pretrained=True)
        self.depth = depth
        print("aaaaaaaaaaaaaa", self.vgg.features)
    
    def forward(self, x):
        return self.vgg.features[:self.depth](x)


# ## ResBlocks

# In[5]:


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, bnorm=nn.BatchNorm2d, relu = nn.LeakyReLU):
        super().__init__()

        if out_channel % 4 != 0:
            raise Exception('ResBlock output should divide by 4')
        
        bottle = int(out_channel//4)
        self.relu = relu
        self.bnorm = bnorm
        self.reluF = relu()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, bottle, kernel_size=1, bias=False),
            bnorm(bottle),
            relu(),

            nn.Conv2d(bottle, bottle, kernel_size=3, padding=1, bias=False, stride=stride),
            bnorm(bottle),
            relu(),

            nn.Conv2d(bottle, out_channel, kernel_size=1, bias=False),
            bnorm(out_channel),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False, stride=stride),
                bnorm(out_channel),
            )
    
    def forward(self, xR):
        #ResBlockForward
        x = self.net(xR)
#         print(x.size(), xR.size())
        shortcut = self.shortcut(xR)
#         print(shortcut.size())
        x += shortcut
        x = self.reluF(x)
        return x

class ResBlockBackward(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, relu=nn.LeakyReLU, bnorm=nn.BatchNorm2d):
        super().__init__()

        if out_channel % 4 != 0:
            raise Exception('ResBlock output should divide by 4')

        bottle = int(out_channel/4)
        self.relu = relu
        self.bnorm = bnorm
        self.reluF = relu()
        self.net = nn.Sequential(
            nn.Conv2d(out_channel, bottle, kernel_size=1, bias=False),
            bnorm(bottle),
            relu(),
            
            nn.Conv2d(bottle, bottle, kernel_size=3, bias=False, padding=1),
            bnorm(bottle),
            relu(),
            nn.UpsamplingNearest2d(scale_factor=stride),
            
            nn.Conv2d(bottle, in_channel, kernel_size=1, bias=False),
            bnorm(in_channel),
        )

        self.shortcut = nn.Sequential()
        if(in_channel != out_channel or stride != 1):
            self.shortcut = nn.Sequential(
                #nn.ConvTranspose2d(out_channel, in_channel, kernel_size=1, stride=stride, bias=False, output_padding=1),
                nn.Conv2d(out_channel, in_channel, kernel_size=1, bias=False),
                bnorm(in_channel),
                nn.UpsamplingNearest2d(scale_factor=stride),
            )
    
    def forward(self, xR):
        x = self.net(xR)
        x += self.shortcut(xR)
        x = self.reluF(x)
        return x


# ## Generator X (Image)
# 
# input: `code[bbox(con), text(con), style(con), latent]` `dim {N, ((5+text_dim)*bbox_count+style_dim), 14, 14}`
# 
# output: `image`

# In[6]:


class GenX(nn.Module):
    def __init__(self, cfg=[3,4,23,8]):
        super(GenX, self).__init__()
        
        def relu():
            return nn.LeakyReLU(0.2, inplace=True)
        def bnorm(ch):
            return nn.BatchNorm2d(ch, 0.8)
        
        def resBlockBackward(repeat, in_channel, out_channel, stride=2):
            layers = []
            for _ in range(repeat-1):
                layers.append(ResBlockBackward(in_channel=out_channel, out_channel=out_channel, stride=1, relu=relu, bnorm=bnorm))
            layers.append(ResBlockBackward(in_channel=in_channel, out_channel=out_channel, stride=stride, relu=relu, bnorm=bnorm))
            return nn.Sequential(*layers)
        
        self.code_dim = calc_code_dim(opt)

        self.net = nn.Sequential(
            nn.BatchNorm2d(self.code_dim),
            #l1
            nn.Conv2d(self.code_dim, 1024, kernel_size=1, bias=False),
            bnorm(1024),
            relu(),
            
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            bnorm(512),
            relu(),

            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            bnorm(1024),
            relu(),
            
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            bnorm(512),
            relu(),
            
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            bnorm(1024),
            relu(),
            
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            bnorm(512),
            relu(),
            
            nn.Conv2d(512, 2048, kernel_size=1, bias=False),
            bnorm(2048),
            relu(),

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
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh(),
        )
    
    def report(self):
        return str(self.net)
    
    def forward(self, bbox, text, style, latent):
        x = torch.cat((bbox, text, style, latent), 1)
        x = self.net(x)
        return x


# ## Generator Z (Code)
# 
# input: `image`
# 
# output: `code[bbox(con), text(con), style(con), latent]`

# In[7]:


class GenZ(nn.Module):
    def __init__(self, cfg=[3,4,23,8]):
        super(GenZ, self).__init__()
        
        def relu():
            return nn.LeakyReLU(0.2, inplace=True)
        def bnorm(ch):
            return nn.BatchNorm2d(ch, 0.8)
        
        def resBlock(repeat, in_channel, out_channel, stride=2):
            layers = []
            layers.append(ResBlock(in_channel=in_channel, out_channel=out_channel, stride=stride, relu=relu, bnorm=bnorm))
            for _ in range(repeat-1):
                layers.append(ResBlock(in_channel=out_channel, out_channel=out_channel, stride=1, relu=relu, bnorm=bnorm))
            return nn.Sequential(*layers)
        
        self.code_dim = calc_code_dim(opt)    

        self.net = nn.Sequential(
            nn.BatchNorm2d(3),
            #downsample 112 < 448
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            bnorm(64),
            relu(),
            nn.MaxPool2d(2),
            
            #resblock cfg[0]=3
            resBlock(cfg[0], 64, 256, stride=1),
            
            #downsample 56 < 112
            #resblock cfg[1]=4
            resBlock(cfg[1], 256, 512),
            
            #downsample 28 < 56
            #resblock cfg[2]=23
            resBlock(cfg[2], 512, 1024),

            #downsample 14 < 28
            #resblock cfg[3]=8
            resBlock(cfg[3], 1024, 2048),
            
            #l1
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            bnorm(512),
            relu(),
            
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            bnorm(1024),
            relu(),
            
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            bnorm(512),
            relu(),
            
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            bnorm(1024),
            relu(),
            
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            bnorm(512),
            relu(),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            bnorm(1024),
            relu(),
            
            nn.Conv2d(1024, self.code_dim, kernel_size=1, bias=False),
            # nn.BatchNorm2d(self.code_dim, 0.8),
            #nn.LeakyReLU(inplace=True),
            nn.Tanh(),
        )
    
    def report(self):
        return str(self.net)
    
    def forward(self, x):
        x = self.net(x)
        bbox = x[:,0:5*opt.bbox_count,:,:]
        text = x[:,5*opt.bbox_count:(5+opt.text_dim)*opt.bbox_count,:,:]
        style = x[:,(5+opt.text_dim)*opt.bbox_count:(5+opt.text_dim+opt.style_dim)*opt.bbox_count,:,:]
        latent = x[:,(5+opt.text_dim+opt.style_dim)*opt.bbox_count:self.code_dim,:,:]
        return bbox, text, style, latent


# ## Discriminator X (Image)
# InfoGAN is applied.
# 
# input: `image`
# 
# output: `valid(1), bbox(5,con), text(opt.text_dim,con), style(opt.style_dim,con)`

# In[8]:


class DiscX(nn.Module):
    def __init__(self):
        super(DiscX, self).__init__()
        
        def relu():
            return nn.LeakyReLU(0.2, inplace=True)
        def bnorm(ch):
            return nn.BatchNorm2d(ch, 0.8)
        
        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        relu(),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(bnorm(out_filters))
            return nn.Sequential(*block)

        self.net_front = nn.Sequential(
            discriminator_block(3, 32, bn=False),
            discriminator_block(32, 64),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**5
        last_ch = 512

        # Output layers
        self.net_valid = nn.Sequential(
            Flat(),
            nn.Linear(last_ch*ds_size**2, 1)
        )
        self.net_bbox = nn.Sequential(
            nn.Conv2d(last_ch, opt.bbox_count * 5, kernel_size=1)
        )
        self.net_text = nn.Sequential(
            nn.Conv2d(last_ch, opt.text_dim * opt.bbox_count, kernel_size=1)
        )
        self.net_style = nn.Sequential(
            nn.Conv2d(last_ch, opt.style_dim * opt.bbox_count, kernel_size=1)
        )

    def forward(self, x):
        x = self.net_front(x)
        valid = self.net_valid(x)
        bbox = self.net_bbox(x)
        text = self.net_text(x)
        style = self.net_style(x)
        return valid, bbox, text, style


# ## Discriminator Z (Code)
# 
# input: `code`
# 
# output: `valid(1)`

# In[9]:


class DiscZ(nn.Module):
    def __init__(self):
        super(DiscZ, self).__init__()
        
        def relu():
            return nn.LeakyReLU(0.2, inplace=True)
        def bnorm(ch):
            return nn.BatchNorm2d(ch, 0.8)
        
        dim = calc_code_dim(opt)
        ds_size = opt.img_size // 2**5

        self.net = nn.Sequential(
            nn.Conv2d(dim, 256, kernel_size=1),
            relu(),
            bnorm(256),

            nn.Conv2d(256, 128, kernel_size=1),
            relu(),
            bnorm(128),

            Flat(),
            
            nn.Linear(128*ds_size**2, 1)
        )
    
    def forward(self, bbox, text, style, latent):
        if text is None and style is None and latent is None:
            x=bbox
        else:
            x=torch.cat((bbox, text, style, latent), 1)
        return self.net(x)

#**note** Disc Opt Rule: MSE to 0(fake) MSE to 1(real)


# # Runtime
# ## Import Dataset

# In[10]:


if __name__ == '__main__':
    #=====================================================
    # load data
    #=====================================================
    if opt.load_data and os.path.exists('latium.data.cache'):
        print('read data cache...')
        with open('latium.data.cache', 'rb') as f:
            data = cPickle.load(f)
    else:
        print('write data cache...')
        data = coco_text_dataset(                      batchW=opt.img_size, batchH=opt.img_size,
                     outputGridW=opt.img_size//2**5, outputGridH=opt.img_size//2**5,
                     catIds=[('legibility','legible')],
                     minSize=[0,14],
                     textDim=opt.text_dim, styleDim=opt.style_dim, latentDim=opt.latent_dim,
                     bboxCount=opt.bbox_count, ignoreCache=False)
        with open('latium.data.cache', 'wb') as f:
            cPickle.dump(data, f)


# ## Training

# In[2]:


#=========================================================
# main program
#=========================================================
if __name__ == "__main__":
    #init logger
    logger = Logger()
    logger.print('logger inited')
    
    #=====================================================
    # init data
    #=====================================================
    
    load_model = opt.load and os.path.exists('latium.model')
    logger.print('initializing...')
    #models
    genX = GenX(cfg=opt.cfg_genX)
    genZ = GenZ(cfg=opt.cfg_genZ)
    discX = DiscX()
    discZ = DiscZ()
    if load_model:
        logger.print('load from disk')
        state = torch.load('latium.model')
        #models
        genX.load_state_dict(state['genX'])
        genZ.load_state_dict(state['genZ'])
        discX.load_state_dict(state['discX'])
        discZ.load_state_dict(state['discZ'])
    if opt.cuda:
        for i in [genX, genZ, discX, discZ]:
            i.cuda()
    logger.print('genX')
#     logger.print(genX.report())
    logger.print('genZ')
#     logger.print(genZ.report())
    #vgg
    logger.print('load vgg')
    featureNet = VGG16Feature(depth = opt.vgg_depth)
    featureNet.cuda()
    featureNet.eval()
    #buffers
    fake_X_buffer = ReplayBuffer()
    fake_Z_buffer = ReplayBuffer()
    #optimizers
    optCycle = torch.optim.Adam(itertools.chain(genX.parameters(), genZ.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    optDz = torch.optim.Adam(discZ.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optDx = torch.optim.Adam(discX.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optInfo = torch.optim.Adam(itertools.chain(genX.parameters(), discX.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
#     optBbox = torch.optim.Adam(genZ.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    #trian
    train_step = 0
    logger.logdir = os.path.join(r"F:\Library\ocr\temp", getTimeStamp())
    
    if load_model:
        #buffers
        fake_X_buffer = state['fake_X_buffer']
        fake_Z_buffer = state['fake_Z_buffer']
#         if opt.cuda:
#             fake_Z_buffer.to(device_cuda)
#             fake_X_buffer.to(device_cuda)
        #optimizers
        optCycle.load_state_dict(state['optCycle'])
        optDz.load_state_dict(state['optDz'])
        optDx.load_state_dict(state['optDx'])
        optInfo.load_state_dict(state['optInfo'])
#         optBbox.load_state_dict(state['optBbox'])
        #trian
        train_step = state['train_step']
        logger.logdir = state['logger_logdir']
    else:
        for i in [genX, genZ, discX, discZ]:
            i.apply(weights_init_normal)
#     discX = DiscX()
#     discX.apply(weights_init_normal)
#     optCycle = torch.optim.Adam(itertools.chain(genX.parameters(), genZ.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
#     optDx = torch.optim.Adam(discX.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#     logger.print('disc x cleared!')
    if opt.lr_update:
        logger.print('update learning rate')
        for o in [optCycle, optDz, optDz, optInfo]:
            for param_group in o.param_groups:
                param_group['lr'] = opt.lr
    for i in [genX, genZ, discX, discZ]:
        i.train()
    if load_model:
        del state
    logger.print('model inited')
    
    #=====================================================
    # init train
    #=====================================================
    
    #losses
    lossGAN = torch.nn.MSELoss()
    lossCycle = torch.nn.L1Loss()
    lossIdentity = torch.nn.L1Loss()
    lossInfoCon = nn.MSELoss()
    lossInfoCat = nn.CrossEntropyLoss()
#     lossMSE = nn.MSELoss()
    #lambdas
    lambda_cat = 1
    lambda_con = 0.1
    target_real = Variable(Tensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)
    #move to cuda
    if opt.cuda:
        for i in [lossCycle, lossInfoCat, lossInfoCon, lossGAN]:
            i.cuda()
    time_log = 0
    time_all = 0
    time_last_img = 0
    time_last_save = 0
    batch_thread = util.ThreadBuffer()
    def batchGet(count):
        batch = data.batch(count)
        batch_image = Tensor(batch['img'])
        batch_image = batch_image.permute(0, 3, 1, 2)
        batch_image = (batch_image.float() - 127.5)/127.5
        batch_bbox = Tensor(batch['bbox'])
        batch_text = Tensor(batch['text'])
        batch_style = Tensor(batch['style'])
        batch_latent = Tensor(batch['latent'])
        return batch, batch_image, batch_bbox, batch_text, batch_style, batch_latent
    logger.print('values inited')
    
    #=====================================================
    # train loop
    #=====================================================
    loss_D_Z = loss_bbox = loss_info = loss_D_X = loss_G = None
    forward_step = 0
    gan_forward_step = 0
    while True:
        log_dict = {}
        #=================================================
        # Batch: Read Data
        #=================================================
        
        s_all = s = time.time()
        
        update_parameter = forward_step != 0 and forward_step % opt.batch_update == 0
        if update_parameter:
            logger.print("update parameter")
        else:
            logger.print("forwarding... {}/{}".format(forward_step%opt.batch_update, opt.batch_update))
        forward_step += 1
        
        for _ in range(opt.gan_update):
            s_batch = time.time()
            
            update_gan = gan_forward_step != 0 and gan_forward_step % opt.batch_update == 0
#             if update_gan:
#                 logger.print("update gan")
#             else:
#                 logger.print("forward gan")
            gan_forward_step += 1
            
            batch, batch_image, batch_bbox, batch_text, batch_style, batch_latent =                 batch_thread.get(batchGet, [opt.batch_size])

            time_batch = time.time()-s_batch

            #=================================================
            # Cycle: Train Generator
            # https://github.com/aitorzip/PyTorch-CycleGAN/
            #=================================================
            
            #==== Identity loss
            # should skipped?

            #==== GAN loss
            fake_Z_bbox, fake_Z_text, fake_Z_style, fake_Z_latent = genZ(batch_image)
            pred_fake = discZ(fake_Z_bbox, fake_Z_text, fake_Z_style, fake_Z_latent)
            loss_GAN_X2Z = lossGAN(pred_fake, target_real)

            fake_X_now = fake_X = genX(batch_bbox, batch_text, batch_style, batch_latent)
            pred_fake, _, _, _ = discX(fake_X)
            loss_GAN_Z2X = lossGAN(pred_fake, target_real)

            #==== Cycle loss
            recovered_X = genX(fake_Z_bbox, fake_Z_text, fake_Z_style, fake_Z_latent)
            recovered_X_feature = featureNet(recovered_X)
            batch_image_feature = featureNet(batch_image)
            loss_cycle_XZX_content = torch.mean(torch.pow(recovered_X_feature-batch_image_feature, 2)) * 1
#             print('hello', recovered_X_feature.size(), loss_cycle_XZX_content)
            loss_cycle_XZX = lossCycle(recovered_X, batch_image) * 10.0

            recovered_Z = genZ(fake_X)
            loss_cycle_ZXZ = lossCycle(torch.cat(recovered_Z, 1), torch.cat((batch_bbox, batch_text, batch_style, batch_latent), 1))*10.0
            
            #==== Bbox loss
            batch_prob_mask = batch_bbox[:,4:5,:,:]
            batch_prob_noobj_mask = (batch_prob_mask - 1) * (-1)
            fake_Z_prob_mask = fake_Z_bbox[:,4:5,:,:]
            loss_bbox_obj = torch.sum(torch.pow(batch_prob_mask - fake_Z_prob_mask, 2) *                                 batch_prob_mask)
            loss_bbox_noobj = 0.5 * torch.sum(torch.pow(batch_prob_mask - fake_Z_prob_mask, 2) *                                        batch_prob_noobj_mask)

            fake_Z_bbox_loc = fake_Z_bbox[:,0:2,:,:]
            batch_bbox_loc = batch_bbox[:,0:2,:,:]
            loss_bbox_loc_mse = 5 * torch.sum(torch.pow(batch_bbox_loc-fake_Z_bbox_loc, 2) *                                  batch_prob_mask)

            fake_Z_bbox_size = fake_Z_bbox[:,2:4,:,:]
            batch_bbox_size = batch_bbox[:,2:4,:,:]
            loss_bbox_size_mse = 5 * torch.sum(torch.pow(batch_bbox_size-                                                         fake_Z_bbox_size, 2) * batch_prob_mask)
            
            #TODO: need add text loss

            loss_bbox = loss_bbox_obj + loss_bbox_noobj + loss_bbox_loc_mse + loss_bbox_size_mse
            
            #==== Total loss
            _loss_G = loss_GAN_X2Z + loss_GAN_Z2X + loss_cycle_XZX + loss_cycle_ZXZ + loss_cycle_XZX_content + loss_bbox
            if loss_G is None:
                loss_G = _loss_G
#                 print('push lossG')
            else:
                _loss_G.data += loss_G.data
                loss_G = _loss_G
#                 print('stack lossG')
            
            optCycle.zero_grad()
            
            if update_gan:
                loss_G.data /= opt.batch_update
                loss_G_scalar = loss_G.item()
#                 log_dict['loss/loss_G'] = loss_G_scalar
            loss_G.backward()
            
            if update_gan:
                optCycle.step()
                loss_G = None
#                 print('claer loss_G')

        time_cycle = time.time()-s

        #=================================================
        # Cycle: train Disc X
        #=================================================
        
        s = time.time()
        
        # Real loss
        pred_real, _, _, _ = discX(batch_image)
        loss_D_real = lossGAN(pred_real, target_real)

        # Fake loss
        fake_X = fake_X_buffer.push_and_pop(fake_X.detach())
        pred_fake, _, _, _ = discX(fake_X)
        loss_D_fake = lossGAN(pred_fake, target_fake)

        # Total loss
        _loss_D_X = (loss_D_real + loss_D_fake)*0.5
        if loss_D_X is None:
            loss_D_X = _loss_D_X
        else:
            _loss_D_X.data += loss_D_X.data
            loss_D_X = _loss_D_X

        optDx.zero_grad()

        if update_parameter:
            loss_D_X.data /= opt.batch_update
        loss_D_X.backward()

        if update_parameter:
            optDx.step()
        
        time_discX = time.time() - s

        #=================================================
        # Cycle: train Disc Z
        #=================================================
        
        s = time.time()

        # Real loss
        pred_real = discZ(batch_bbox, batch_text, batch_style, batch_latent)
        loss_D_real = lossGAN(pred_real, target_real)
        
        # Fake loss
        fake_Z_code = fake_Z_buffer.push_and_pop(            torch.cat((fake_Z_bbox.detach(), fake_Z_text.detach(), fake_Z_style.detach(), fake_Z_latent.detach()), 1))
        pred_fake = discZ(fake_Z_code, None, None, None)
        loss_D_fake = lossGAN(pred_fake, target_fake)

        # Total loss
        _loss_D_Z = (loss_D_real + loss_D_fake)*0.5
        if loss_D_Z is None:
            loss_D_Z = _loss_D_Z
        else:
            _loss_D_Z.data += loss_D_Z.data
            loss_D_Z = _loss_D_Z

        optDz.zero_grad()

        if update_parameter:
            loss_D_Z.data /= opt.batch_update
        loss_D_Z.backward()

        if update_parameter:
            optDz.step()
        
        time_discZ = time.time() - s
        
        #=================================================
        # Info: train Info GenX
        #=================================================
        
        s = time.time()
        
        optInfo.zero_grad()
        
        # Sample noise, labels and code as generator input
        _, pred_bbox, pred_text, pred_style = discX(genX(batch_bbox, batch_text, batch_style, batch_latent))
        
        _loss_info = lambda_con * (lossInfoCon(pred_bbox, batch_bbox) +                                   lossInfoCon(pred_text, batch_text) +                                   lossInfoCon(pred_style, batch_style)) * 0.33

        if loss_info is None:
            loss_info = _loss_info
        else:
            _loss_info.data += loss_info.data
            loss_info = _loss_info

        optInfo.zero_grad()

        if update_parameter:
            loss_info.data /= opt.batch_update
        loss_info.backward()

        if update_parameter:
            optInfo.step()
        
        time_info = time.time() - s
        
        #=================================================
        # YOLO: train BBox
        #=================================================
        
        s = time.time()
        
#         fake_Z_bbox, fake_Z_text, fake_Z_style, fake_Z_latent = genZ(batch_image)
        
#         batch_prob_mask = batch_bbox[:,4:5,:,:]
#         batch_prob_noobj_mask = (batch_prob_mask - 1) * (-1)
#         fake_Z_prob_mask = fake_Z_bbox[:,4:5,:,:]
#         loss_bbox_obj = torch.sum(torch.pow(batch_prob_mask - fake_Z_prob_mask, 2) * \
#                             batch_prob_mask)
#         loss_bbox_noobj = 0.5 * torch.sum(torch.pow(batch_prob_mask - fake_Z_prob_mask, 2) *\
#                                     batch_prob_noobj_mask)
        
#         fake_Z_bbox_loc = fake_Z_bbox[:,0:2,:,:]
#         batch_bbox_loc = batch_bbox[:,0:2,:,:]
#         loss_bbox_loc_mse = 5 * torch.sum(torch.pow(batch_bbox_loc-fake_Z_bbox_loc, 2) * \
#                              batch_prob_mask)
        
#         fake_Z_bbox_size = fake_Z_bbox[:,2:4,:,:]
#         batch_bbox_size = batch_bbox[:,2:4,:,:]
#         loss_bbox_size_mse = 5 * torch.sum(torch.pow(torch.sqrt(torch.abs(batch_bbox_size))-\
#                                                      torch.sqrt(torch.abs(fake_Z_bbox_size)), 2) * batch_prob_mask)
        
#         #TODO: need add text loss
        
#         _loss_bbox = loss_bbox_obj + loss_bbox_noobj + loss_bbox_loc_mse + loss_bbox_size_mse
#         if loss_bbox is None:
#             loss_bbox = _loss_bbox
#         else:
#             _loss_bbox.data += loss_bbox.data
#             loss_bbox = _loss_bbox

#         optBbox.zero_grad()

#         if update_parameter:
#             loss_bbox.data /= opt.batch_update
#         loss_bbox.backward()

#         if update_parameter:
#             optBbox.step()
        
        time_bbox = time.time() - s
        
        #=================================================
        # Logging
        #=================================================        
        
        s = time.time()
        
        if update_parameter:
            def postImg(batch, i=0, cvtBGR=False):
                img = batch[i].detach()
                img = img * 127.5 + 127.5
                img = torch.clamp(img,0,255)
                img = img.permute(1,2,0).cpu().numpy().astype(np.uint8)
                img = cv2.resize(img, dsize=(opt.img_size, opt.img_size))
                if cvtBGR:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            if in_notebook() or time.time()-time_last_img > opt.pic_interval:
                real_img = postImg(batch_image)
                fake_img = postImg(fake_X_now)
                recv_img = postImg(recovered_X)
                img = np.concatenate((real_img, fake_img, recv_img), axis=1)
                if in_notebook():
                    display.vidshow(img, maxSize=(800,400))
                logger.log_image({                        'img/real_fake_recv' : cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        }, global_step=train_step)
                if ((time_last_img == 0 and train_step == 0 and (not os.path.exists('latium.model'))) or train_step != 0) and                     not opt.no_save and time.time()-time_last_save > opt.save_interval:
                    logger.print('saving...')
                    state = {
                        'genX':genX.state_dict(),
                        'genZ':genZ.state_dict(),
                        'discX':discX.state_dict(),
                        'discZ':discZ.state_dict(),
                        'fake_X_buffer':fake_X_buffer,
                        'fake_Z_buffer':fake_Z_buffer,
                        'optCycle':optCycle.state_dict(),
                        'optDz':optDz.state_dict(),
                        'optDx':optDx.state_dict(),
                        'optInfo':optInfo.state_dict(),
#                             'optBbox':optBbox.state_dict(),
                        'train_step':train_step,
                        'logger_logdir':logger.logdir,
                    }
                    torch.save(state, 'latium.model')
                    logger.print("saved")
                    del state
                    time_last_save = time.time()
                time_last_img = time.time()

            logger.log({                        'loss/loss_G':loss_G_scalar, 
                        'loss/GAN/X2Z':loss_GAN_X2Z, 
                        'loss/GAN/Z2X':loss_GAN_Z2X,
                        'loss/GAN/cycleXZX':loss_cycle_XZX,
                        'loss/GAN/cycleZXZ':loss_cycle_ZXZ,
                        'loss/GAN/cycleXZXcontent':loss_cycle_XZX_content,
                        'loss/loss_D_Z':loss_D_Z,
                        'loss/loss_D_X':loss_D_X,
                        'loss/loss_Info':loss_info,
                        'loss/loss_bbox':loss_bbox,
                        'loss/bbox/loc':loss_bbox_loc_mse,
                        'loss/bbox/noobj':loss_bbox_noobj,
                        'loss/bbox/obj':loss_bbox_obj,
                        'loss/bbox/size':loss_bbox_size_mse,
                        'time/cycle':time_cycle,
                        'time/batch':time_batch,
                        'time/discX':time_discX,
                        'time/discZ':time_discZ,
                        'time/info':time_info,
                        'time/bbox':time_bbox,
                        'time/log':time_log,
                        'time/all':time_all,
                       }, global_step=train_step)
            train_step += 1
            loss_D_X = loss_D_Z = loss_info = loss_bbox = None
            time_log = time.time() - s
        del log_dict
        time_all = time.time() - s_all


# ## Export

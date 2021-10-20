
#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami, Sheng Shen
# All rights reserved.
# This file is part of AdaHessian library.
#
# AdaHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AdaHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with adahessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function
import logging
import os
import sys

import numpy as np
import argparse
from tqdm import tqdm, trange

from SGDHess import SGDHess
from sgd import sgd, sgd_Snapshot
from STORM import STORM
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
from models.resnet import *
from optim_adahessian import Adahessian

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch-size', type=int, default= 256, metavar='B',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default = 256, metavar='TB',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.15, metavar='LR',
                    help='learning rate (default: 0.15)')
parser.add_argument('--c', type=float, default=100, metavar='C',
                    help='momentum constant (default: 100)')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='learning rate ratio')
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[80, 120],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--depth', type=int, default=20,
                    help='choose the depth of resnet')
parser.add_argument('--optimizer', type=str, default='adahessian',
                    help='choose optim')

args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


for arg in vars(args):
    print(arg, getattr(args, arg))
if not os.path.isdir('checkpoint/'):
    os.makedirs('checkpoint/')
# get dataset
train_loader, test_loader = getData(
    name='cifar10', train_bs=args.batch_size, test_bs=args.test_batch_size)

# make sure to use cudnn.benchmark for second backprop
cudnn.benchmark = True

# get model and optimizer
model = resnet(num_classes=10, depth=args.depth).cuda()
print(model)
model = torch.nn.DataParallel(model)
print('    Total params: %.2fM' % (sum(p.numel()
                                       for p in model.parameters()) / 1000000.0))

criterion = nn.CrossEntropyLoss()
if args.optimizer == 'sgd':
    optimizer = sgd( model.parameters(),
        lr=args.lr,
        momentum=0.9,
      weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    print('For AdamW, we automatically correct the weight decay term for you! If this is not what you want, please modify the code!')
    args.weight_decay = args.weight_decay / args.lr
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
elif args.optimizer == 'adahessian':
    print('For AdaHessian, we use the decoupled weight decay as AdamW. Here we automatically correct this for you! If this is not what you want, please modify the code!')
    args.weight_decay = args.weight_decay / args.lr
    optimizer = Adahessian(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
elif args.optimizer == 'sgdhess':
    print('SGDHess')
    args.weight_decay = args.weight_decay
    optimizer = SGDHess(
        model.parameters(),
        lr=args.lr,
        momentum = 0.9,
        weight_decay=args.weight_decay)
elif args.optimizer == 'storm':
    optimizer = STORM(model.parameters(), lr = args.lr, c = args.c)
else:
    raise Exception('We do not support this optimizer yet!!')

# learning rate schedule
# Don't use for STORM
scheduler = lr_scheduler.MultiStepLR(
    optimizer,
    args.lr_decay_epoch,
    gamma=args.lr_decay,
  last_epoch=-1)
best_acc = 0.0
accuracy = []
L = []
step = 0
for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    max_L = 0
    scheduler.step()
    model.train()
    with tqdm(total=len(train_loader.dataset)) as progressbar:
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if (args.optimizer == 'storm'):
                data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward(create_graph=True)
                if  step == 0:
                    optimizer.update_momentum()
                    step += 1
                else:
                    optimizer.clone_grad()
                    optimizer.zero_grad()
                    optimizer.step()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.update_momentum()
            else:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward(create_graph=True)
                params1,grads1 = optimizer.get_params()
                optimizer.step()
                output1 = model(data)
                loss1 = criterion(output1, target)
                optimizer.zero_grad()
                loss1.backward(create_graph=True)
                params2, grads2 = optimizer.get_params()
                for j in range(1, len(params1)):
                    temp_L = torch.norm(grads1[j] - grads2[j])/torch.norm(params1[j] - params2[j])
                    if(temp_L >= max_L):
                        max_L = temp_L
                 
            train_loss += loss.item() * target.size()[0]
            total_num += target.size()[0]
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            progressbar.update(target.size(0))

    acc = test(model, test_loader)
    train_loss /= total_num
    print(f"Training Loss of Epoch {epoch}: {np.around(train_loss, 2)}")
    print(f"Testing of Epoch {epoch}: {np.around(acc * 100, 2)} \n")
    print("current max L is", max_L)
    L.append(max_L)
    accuracy.append(acc)
    if acc > best_acc:
        best_acc = acc
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_accuracy': best_acc,
            }, 'checkpoint/netbest.pkl')

print(f'Best Acc: {np.around(best_acc * 100, 2)}')

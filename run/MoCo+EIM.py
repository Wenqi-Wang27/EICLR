#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import random
import logging
import argparse
import builtins
import datetime
import warnings
from PIL import ImageFilter

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torchvision import transforms as TF
from model.resnet18 import MoCo


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MoCo Training')
parser.add_argument('--data_sort', type=str, help="option for data_type; 'UIUC_Sports', 'NEU_CLS','MSTAR'")
parser.add_argument('--pre_data', type=str, help='path to dataset')
parser.add_argument('--temperature', type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--log_dir', type=str, help='path to save state_dict')
parser.add_argument('--batch_size', type=int, help='mini-batch size (default: 256), this is the total ')
# options for HX
parser.add_argument('--lamda', type=float, help='The weight of Entropy_loss')
parser.add_argument('--momentum', type=float, help='momentum of SGD solver')
parser.add_argument('--wd', type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, help='seed for initializing training. ')
parser.add_argument('--epochs', type=int, help='number of total epochs to run')
parser.add_argument('--lr', type=float, help='initial learning rate', dest='lr')
parser.add_argument('--save_freq', type=int, help='save state_dict frequency (default: 10)')
parser.add_argument('--arch', default='resnet18', type=str, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--workers', type=int, help='number of data loading workers (default: 32)')
parser.add_argument('--start_epoch', type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--weight_decay', type=float, help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='GPU id to use.')
# moco specific configs:
parser.add_argument('--moco_dim', type=int, help='feature dimension (default: 128)')
parser.add_argument('--moco_k', type=int, help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco_m', type=float, help='moco momentum of updating key encoder (default: 0.999)')


def main():
    logging.info(f"args: {args}\t")
    logging.info('Using device {} for training'.format(args.device))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = MoCo(base_encoder=models.__dict__[args.arch], out_dim=args.moco_dim, queue_size=args.moco_k, m=args.moco_m, T=args.temperature, infoNCE_layer=2, HX_layer=2).to(args.device)
    print(model)
    model.to(args.device)

    train_dataset = datasets.ImageFolder(args.pre_data, TwoCropsTransform())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        for batch, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()
            logits, labels, q_out = model(im_q=images[0].to(args.device), im_k=images[1].to(args.device))
            MoCo_loss = criterion(logits, labels.to(args.device))
            E_loss = args.lamda * Entropy_loss_2D(q_out)
            loss = MoCo_loss + E_loss
            loss.backward()
            optimizer.step()
            logging.info('epoch:({}-{}) MoCo_loss: {:.6f} E_loss: {:.6f} loss: {:.6f}'
                         .format(epoch + 1, batch, MoCo_loss, E_loss, loss))

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                            args.log_dir,
                            filename='lamda={}_tau_{}_epoch_{:04d}.pth.tar'.format(args.lamda, args.temperature, epoch + 1))
            logging.info('tau_{}_epoch{:04d}.pth.tar saved!'.format(args.temperature, epoch + 1))

    logging.info("Training has finished.")


def save_checkpoint(state, log_dir, filename):
    filename_path = os.path.join(log_dir, filename)
    torch.save(state, filename_path)
    print(filename + ' has been saved.')

def Entropy_loss_2D(features, epsilon=1e-08):
    P = torch.softmax(features, dim=1)
    H_X = torch.sum((P * (- torch.log2(P + epsilon))), dim=1)
    loss = torch.exp(-torch.mean(H_X))
    return loss

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

def init_logging(filedir:str):
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    logger = logging.getLogger()
    fh = logging.FileHandler(filename= filedir + '/log_' + get_date_str() + '.txt')
    sh = logging.StreamHandler()
    formatter_fh = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    formatter_sh = logging.Formatter('%(message)s')
    fh.setFormatter(formatter_fh)
    sh.setFormatter(formatter_sh)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(10)
    fh.setLevel(10)
    sh.setLevel(10)
    return logging

if __name__ == '__main__':
    args = parser.parse_args()
    logging = init_logging(args.log_dir)
    main()
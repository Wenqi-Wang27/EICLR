# -*-coding:utf-8-*-
import os
import time
import yaml
import shutil
import random
import warnings
import argparse
import logging
import datetime

import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torchvision import transforms as TF

from PIL import Image
from PIL import ImageFilter

from model.resnet18 import model_SimCLR

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR Pretrain')
parser.add_argument('--data_sort', type=str, help="option for data_type; 'UIUC_Sports', 'NEU_CLS', 'MSTAR', '15_Scene'")
parser.add_argument('--pre_data', type=str, help='path to dataset')
parser.add_argument('--temperature', type=float, help='softmax temperature (default: 0.5)')
parser.add_argument('--log_dir', type=str, help='path to save state_dict')
parser.add_argument('--batch_size', type=int, help='mini-batch size (default: 256), this is the total')
# options for HX
parser.add_argument('--lamda', type=float, help='The weight of Entropy_loss')
parser.add_argument('--seed', type=int, help='seed for initializing training. ')
parser.add_argument('--epochs', type=int, help='number of total epochs to run')
parser.add_argument('--lr', type=float, help='initial learning rate', dest='lr')
parser.add_argument('--save_freq', type=int, help='save state_dict frequency (default: 10)')
# options for SimCLR_MLP
parser.add_argument('--linear1_out', type=int, help='feature dimension (default: 512)')
parser.add_argument('--out_dim', type=int, help='feature dimension (default: 128)')
parser.add_argument('--arch', help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--workers', type=int, help='number of data loading workers (default: 32)')
parser.add_argument('--weight_decay', type=float, help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--n_views', type=int, help='Number of views for contrastive learning training.')
parser.add_argument('--gpu_index', type=int, help='Gpu index.')
parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='GPU id to use.')

def main():
    logging.info(f"args: {args}\t")
    logging.info('Using device {} for training'.format(args.device))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    dataset_simclr = dataset_SimCLR(args.pre_data, args.data_sort)
    train_loader = torch.utils.data.DataLoader(dataset_simclr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    model = model_SimCLR(base_model=args.arch, out_dim=128)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)

class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.relu = nn.ReLU()

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        loss_infoNCE = self.criterion(logits, labels)

        return loss_infoNCE

    def train(self, train_loader):
        logging.info(f"args: {self.args}\t")

        for epoch_counter in range(self.args.epochs):
            for batch, (img_i, img_j) in enumerate(train_loader):
                images = torch.cat((img_i, img_j), dim=0)
                images = images.to(self.args.device)
                self.optimizer.zero_grad()
                self.model.backbone.avgpool.register_forward_hook(get_activation('backbone.avgpool'))
                self.model.backbone.fc[0].register_forward_hook(get_activation('backbone.1linear'))
                features_output = self.model(images)

                SimCLR_loss = self.info_nce_loss(features_output)
                E_loss = args.lamda * Entropy_loss_2D(features_output)
                loss = SimCLR_loss + E_loss
                loss.backward()
                self.optimizer.step()
                logging.info('epoch:({}-{}) lr: {:.6f} SimCLR_loss: {:.6f} E_loss: {:.6f} loss: {:.6f}'
                             .format(epoch_counter + 1, batch, self.scheduler.get_lr()[0], SimCLR_loss, E_loss, loss))

            # warmup for the first 10 epochs
            if (epoch_counter + 1) >= 10:
                self.scheduler.step()

            if (epoch_counter + 1) % args.save_freq == 0:
                save_checkpoint({'epoch': epoch_counter + 1, 'arch': args.arch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                                args.log_dir,
                                filename='lamda={}_tau_{}_epoch_{:04d}.pth.tar'.format(args.lamda, args.temperature, epoch_counter + 1))
                logging.info('tau_{}_epoch{:04d}.pth.tar saved!'.format(args.temperature, epoch_counter + 1))

        logging.info("Training has finished.")


def Entropy_loss_2D(features, epsilon=1e-08):
    P = torch.softmax(features, dim=1)
    H_X = torch.sum((P * (- torch.log2(P + epsilon))), dim=1)
    loss = torch.exp(-torch.mean(H_X))
    # loss = 1 / torch.mean(H_X)
    return loss

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output

    return hook

def save_checkpoint(state, log_dir, filename):
    filename_path = os.path.join(log_dir, filename)
    torch.save(state, filename_path)
    print(filename + ' has been saved.')


def init_logging(filedir:str):
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    logger = logging.getLogger()
    fh = logging.FileHandler(filename= filedir + '/log_' + get_date_str() + '.txt')
    sh = logging.StreamHandler()
    formatter_fh = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    formatter_sh = logging.Formatter('%(message)s')
    # formatter_sh = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    fh.setFormatter(formatter_fh)
    sh.setFormatter(formatter_sh)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(10)
    fh.setLevel(10)
    sh.setLevel(10)

    return logging


if __name__ == "__main__":
    args = parser.parse_args()
    logging = init_logging(args.log_dir)
    main()
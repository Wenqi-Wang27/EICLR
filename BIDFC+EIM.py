# -*-coding:utf-8-*-
import argparse
import datetime
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms as TF
from PIL import ImageFilter
from PIL.Image import Image

from model.resnet18 import model_BIDFC

history_parameters = None

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Pre-Training')
parser.add_argument('--data_sort', type=str, help="option for data_type")
parser.add_argument('--pre_data', type=str, help='path to dataset')
parser.add_argument('--temperature', type=float, help='Temperature Coefficient')
parser.add_argument('--log_dir', type=str, help='path to save state_dict')
parser.add_argument('--batch_size', type=int, help='mini-batch size for training')
parser.add_argument('--denominator', type=float, help='lr = lr / denominator')
parser.add_argument('--arch', default='resnet18', help='model architecture (default: resnet18)')
parser.add_argument('--workers', type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, help='number of total iteration to run')
parser.add_argument('--start_epoch', type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, help='initial learning rate')
parser.add_argument('--SGD', type=bool, help='option for optimizer(SGD or Adam)')
parser.add_argument('--momentum', type=float, help='momentum of SGD solver')
parser.add_argument('--weight_decay', type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--save_freq', type=int, help='save state_dict frequency (default: 10)')
parser.add_argument('--resume', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', type=int, help='seed for initializing training. ')
parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='GPU id to use.')
# BDFC specific configs:
parser.add_argument('--augment_times', type=int, help='K times augmentation (default:5)')
parser.add_argument('-T1', type=int, help='A parameter for the weight of DWVLoss')
parser.add_argument('-T2', type=int, help='A parameter for the weight of DWVLoss')
parser.add_argument('-af', type=int, help='A parameter for the weight of DWVLoss')
parser.add_argument('-ema', type=int, help='A parameter for the BIDFC model update')
# options for HX
parser.add_argument('--lamda', type=float, help='The weight of Entropy_loss')

def main():
    logging.info(f"args: {args}\t")
    best_acc = 0.0
    logging.info('Using device {} for training'.format(args.device))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create BIDFC_model
    logging.info('=> creating BIDFC_model')
    model = model_BIDFC(args.arch, args.batch_size).to(args.device)
    model.to(args.device)
    print(model)

    # define loss function (criterion) and optimizer
    CELoss = CrossEntropyLoss().to(args.device)
    DWVLoss = Dyn_Wei_Var_Loss().to(args.device)
    criterion = [CELoss,DWVLoss]

    args.lr = args.lr * args.batch_size / args.denominator
    if args.SGD:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    train_dataset = BDFC_Dataset(root=args.pre_data, data_sort=args.data_sort, aug_times=args.augment_times, batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        trainer(train_loader, model, criterion, optimizer, epoch, args)
        if (epoch+1) % args.save_freq == 0:
            save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                            args.log_dir,
                            filename='lamda={}_tau_{}_epoch_{:04d}.pth.tar'.format(args.lamda, args.temperature, epoch+1))
            print('tau_{}_epoch{:04d}.pth.tar saved!'.format(args.temperature, epoch+1))


def trainer(train_loader,model,criterion,optimizer,epoch,args):
    global history_parameters
    history_parameters = model.parameters()
    model.train()
    CELoss, DWVLoss = criterion[0],criterion[1]

    cls_running_loss = 0.0
    dwv_running_loss = 0.0

    for batch, (img_list,target_list) in enumerate(train_loader):
        nn.init.xavier_normal_(model.backbone.fc.weight)
        img_list = img_list.transpose(1, 0)
        target_list = target_list.transpose(1, 0)

        # K Times Backpropagation for CrossEntropyLoss
        for i in range(args.augment_times):
            permutation = torch.randperm(target_list[i].shape[0])
            data = img_list[i][permutation, :, :, :]
            target = target_list[i][permutation]
            data, target = data.to(args.device), target.to(args.device)

            optimizer.zero_grad()
            out = model(data)
            # print('P:', F.softmax(out/args.temperature, dim=1)[0])
            cls_loss = CELoss(out/args.temperature, target)
            cls_running_loss = cls_loss.item()
            cls_loss.backward()
            optimizer.step()

        # One Time Backpropagation for DWVLoss
        dwv_input = torch.tensor(0).to(args.device)
        optimizer.zero_grad()
        for i in range(args.augment_times):
            data = img_list[i].to(args.device)
            model.backbone.avgpool.register_forward_hook(get_activation('backbone.avgpool'))
            model(data)
            if i == 0:
                dwv_input = torch.unsqueeze(activation['backbone.avgpool'], 1)
            else:
                dwv_input = torch.cat([dwv_input, torch.unsqueeze(activation['backbone.avgpool'], 1)], dim=1)
        dwv_loss = DWV_Weight(epoch,args.af,args.T1,args.T2) * DWVLoss(dwv_input)
        E_loss = args.lamda * Entropy_loss_3D(dwv_input)
        FC_loss = dwv_loss + E_loss
        FC_loss.backward()
        optimizer.step()
        update_ema_variables(model, args.ema)
        logging.info('(epoch: {} / batch: {} / augment_times: {}) cls_loss: {:.6f} dwv_loss: {:.6f} E_loss: {:.6f} FC_loss: {:.6f}'
                     .format(epoch+1, batch + 1, args.augment_times, cls_running_loss, dwv_loss.item(), E_loss.item(), FC_loss.item()))


def Entropy_loss_3D(features, epsilon=1e-08):
    P = torch.softmax(features, dim=2)
    H_X = torch.sum((P * (- torch.log2(P + epsilon))), dim=2)
    loss = torch.exp(-torch.mean(H_X))
    return loss

def CrossEntropyLoss():
    return nn.CrossEntropyLoss()

class Dyn_Wei_Var_Loss(nn.Module):
    def forward(self, x):
        return torch.sum(torch.mean(torch.std(x, dim=1), dim=1))


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook


def DWV_Weight(epoch,af,T1,T2):
    alpha = 0.0
    if epoch > T1:
        alpha = (epoch-T1) / (T2-T1)*af
        if epoch > T2:
            alpha = af
    return alpha

def update_ema_variables(model, alpha):
    # Use the true average until the exponential average is more correct
    global history_parameters
    for history_param, param in zip(history_parameters, model.parameters()):
        param.data = alpha * history_param.data + (1-alpha) * param.data
    history_parameters = model.parameters()


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


if __name__ == '__main__':
    args = parser.parse_args()
    logging = init_logging(args.log_dir)
    main()
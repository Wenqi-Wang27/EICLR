# -*-coding:utf-8-*-
import argparse
import os
import random
import warnings
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
import torchvision.models as models

class model_BIDFC(nn.Module):
    def __init__(self, base_model, out_dim):
        super(model_BIDFC, self).__init__()
        self.out_dim = out_dim
        self.backbone = self.get_resnet(base_model)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def get_resnet(self, name, pretrained=False):
        resnets = {"resnet18": torchvision.models.resnet18(pretrained=pretrained, num_classes=self.out_dim),
                   "resnet50": torchvision.models.resnet50(pretrained=pretrained, num_classes=self.out_dim)}
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]

class MoCo(nn.Module):
    def __init__(self, base_encoder, out_dim:int, queue_size:int, m:float, T:float, infoNCE_layer:int, HX_layer:int):
        super(MoCo, self).__init__()
        self.queue_size = queue_size
        self.m = m
        self.T = T
        self.infoNCE_layer = infoNCE_layer
        self.HX_layer = HX_layer

        # create the encoders
        # num_classes is the output fc dimension
        self.online_network = base_encoder(num_classes=out_dim)
        self.target_network = base_encoder(num_classes=out_dim)

        dim_mlp = self.online_network.fc.weight.shape[1]
        self.online_network.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.online_network.fc)
        self.target_network.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.target_network.fc)

        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(out_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        q_out = self.online_network(im_q)  # queries: NxC


        q = F.normalize(q_out, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k_out = self.target_network(im_k)  # keys: NxC
            k = F.normalize(k_out, dim=1)

        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits = logits / self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, q_out


class model_SimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(model_SimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, 2 * dim_mlp), nn.ReLU(), nn.Linear(2 * dim_mlp, out_dim))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)

class BaseSimCLRException(Exception):
    """Base exception"""

class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""

class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""
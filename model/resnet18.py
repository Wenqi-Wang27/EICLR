# -*-coding:utf-8-*-
import argparse
import os
import random
import warnings
import torch.nn as nn
import torchvision



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
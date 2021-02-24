# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 00:02:03 2021

@author: konqr
"""
import torch 
import torch.nn as nn

class ChessModel(nn.Module):
    
    def __init__(self):
        super(ChessModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(12,32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1))
        self.fc = nn.Linear(20*1024, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out)
        out = self.fc(out)
        return out
        
    
if __name__ == "__main__":
    model = ChessModel()
    ip = torch.randn(20,12,8,8)
    print(model.forward(ip))
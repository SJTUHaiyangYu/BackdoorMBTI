import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaClassifier(nn.Module):
    def __init__(self, input_size, class_num, N_in=10, gpu=False):
        super(MetaClassifier, self).__init__()
        self.input_size = input_size
        self.class_num = class_num
        self.N_in = N_in
        self.N_h = 20
        self.inp = nn.Parameter(torch.zeros(self.N_in, *input_size).normal_() * 1e-3)
        self.fc = nn.Linear(self.N_in * self.class_num, self.N_h)
        self.output = nn.Linear(self.N_h, 1)

        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def forward(self, pred):
        emb = F.relu(self.fc(pred.view(self.N_in * self.class_num)))
        score = self.output(emb)
        return score

    def loss(self, score, y):
        y_var = torch.FloatTensor([y])
        if self.gpu:
            y_var = y_var.cuda()
        l = F.binary_cross_entropy_with_logits(score, y_var)
        return l

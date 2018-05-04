import torch
import torch.nn as nn
from math import sqrt
import numpy as np


class FullConnect(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullConnect, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        u = 1 / sqrt(input_size)
        torch.nn.init.uniform_(self.weight, -u, u)
        self.bias.data.zero_()

    def forward(self, x):
        logit = torch.matmul(x, self.weight) + self.bias
        return logit


class LstmCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_param()

    def forward(self, x, h_fore, c_fore):
        input = torch.cat([h_fore, x], dim=-1)
        i, f, o, c = torch.matmul(input, self.Wi) + self.Bi, torch.matmul(input, self.Wf) + self.Bf, \
                     torch.matmul(input, self.Wo) + self.Bo, torch.matmul(input, self.Wc) + self.Bc
        i, f, o, c = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(c)
        c = f * c_fore + i * c
        h = o * torch.tanh(c)
        return h, c

    def init_param(self):
        u = 1 / sqrt(self.hidden_size)
        dim_param = self.input_size + self.hidden_size
        self.Wi = nn.Parameter(torch.Tensor(dim_param, self.hidden_size))
        self.Wf = nn.Parameter(torch.Tensor(dim_param, self.hidden_size))
        self.Wo = nn.Parameter(torch.Tensor(dim_param, self.hidden_size))
        self.Wc = nn.Parameter(torch.Tensor(dim_param, self.hidden_size))
        self.Bi = nn.Parameter(torch.Tensor(self.hidden_size))
        self.Bf = nn.Parameter(torch.Tensor(self.hidden_size))
        self.Bo = nn.Parameter(torch.Tensor(self.hidden_size))
        self.Bc = nn.Parameter(torch.Tensor(self.hidden_size))
        torch.nn.init.uniform_(self.Wi, -u, u)
        torch.nn.init.uniform_(self.Wf, -u, u)
        torch.nn.init.uniform_(self.Wo, -u, u)
        torch.nn.init.uniform_(self.Wc, -u, u)
        self.Bi.data.zero_()
        self.Bf.data.zero_()
        self.Bo.data.zero_()
        self.Bc.data.zero_()


def get_mask(maxlen, lens):
    # lens, cpu tensor
    batch = lens.shape[0]
    index = np.arange(maxlen).repeat(batch).reshape(maxlen, batch).transpose([1, 0])
    mask = index < lens[:, None].numpy()
    mask = mask.astype(np.float)
    return torch.from_numpy(mask).float()


def get_acc(logit, labels):
    correct = torch.sum(torch.argmax(logit, dim=-1) == labels)
    acc = correct.float() / len(labels)
    return acc


def masked_softmax(A, mask):
    # matrix A is the one you want to do mask softmax at dim=1
    A_max = torch.max(A, dim=1, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * mask  # this step masks
    A_softmax = A_exp / (torch.sum(A_exp, dim=1, keepdim=True) + 1e-10)
    return A_softmax


if __name__ == '__main__':
    logit = torch.randn(5, 3)
    labels = torch.from_numpy(np.array([0, 0, 0, 0, 0])).long()
    print(logit)
    print(labels)
    print(get_acc(logit, labels))

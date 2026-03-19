import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image


class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab

class GradientReconstructor:
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std, config, num_images=1):
        self.model = model
        self.model.flag = 1
        self.config = config
        self.num_images = num_images
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        self.mean_std = mean_std
        self.loss_fn = nn.CrossEntropyLoss().to(**self.setup)
        self.parameters = [p for name, p in self.model.named_parameters() if 'eps' not in name]

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), num_classes=1, dryrun=False, eval=True, tol=None):
        start_time = time.time()  # 记录运算开始时刻
        if eval:
            self.model.eval()

        ''' train DLG and iDLG '''
        criterion = self.loss_fn
        original_dy_dx = list((_.detach().clone() for _ in input_data))
        gt_label = labels

        # generate dummy data and label
        dummy_data = self._init_images(img_shape).requires_grad_(True)
        dummy_label = torch.randn((self.num_images, num_classes)).to(**self.setup).requires_grad_(True)

        if self.config['method'] == 'DLG':
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=self.config['lr'])
        elif self.config['method'] == 'iDLG':
            optimizer = torch.optim.LBFGS([dummy_data, ], lr=self.config['lr'])
            # predict the ground-truth label
            label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
        
        Iteration = self.config['max_iterations']
        losses = []
        train_iters = []
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[Iteration // 2.667, Iteration // 1.6,
                                                                         Iteration // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        for iters in range(Iteration):

            def closure():
                optimizer.zero_grad()
                self.model.zero_grad()
                pred = self.model(dummy_data)
                if self.config['method'] == 'DLG':
                    dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1)) + self.model.loss()
                    # dummy_loss = criterion(pred, gt_label)
                elif self.config['method'] == 'iDLG':
                    dummy_loss = criterion(pred, gt_label) + self.model.loss()

                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.parameters, create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()

                if self.config['signed']:
                    dummy_data.grad.sign_()

                return grad_diff

            optimizer.step(closure)
            if self.config['lr_decay']:
                scheduler.step()

            current_loss = closure().item()
            train_iters.append(iters)
            losses.append(current_loss)

            if self.config['boxed']:
                dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

            if iters % int(Iteration / 30) == 0:
                current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                print(current_time, iters, 'loss = %.8f' %(current_loss))

                if current_loss < 0.000001: # converge
                    break

        if self.config['method'] == 'DLG':
            loss_DLG = losses
            print('loss_DLG:', loss_DLG[-1])
        elif self.config['method'] == 'iDLG':
            loss_iDLG = losses
            print('loss_iDLG:', loss_iDLG[-1])

        time_run = time.time() - start_time
        print(f'Run time: {time_run}.')  # 打印运行时间
        return dummy_data.detach(), dummy_label.detach()

    def _init_images(self, img_shape): #随机初始化图像张量
        if self.config['init'] == 'randn': #标准正态分布N(0,1)
            return torch.randn((self.num_images, *img_shape), **self.setup) #随机生成(1*1*3*32*32)张量
        elif self.config['init'] == 'rand': #均匀分布[0,1]
            return (torch.rand((self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros': #全部置零
            return torch.zeros((self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()


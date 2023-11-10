import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
import numpy as np
import os

from dataset import Img_dataset
from posenet_resnet50 import PoseNet

# Fix Seed ------------------------------------------------
import random
seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Machine is',device)

# Config ------------------------------------------------
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='PoseNet')
    parser.add_argument('--trainset_dir', type=str, default='./dataset/training/')
    parser.add_argument('--validset_dir', type=str, default='./dataset/validation/')
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=50, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument('--model_path', type=str, default='best_model/best_model_130.pt')

    return parser.parse_args()


# Test ------------------------------------------------


def test_vis(cfg):
    cudnn.benchmark = True

    net = PoseNet()
    net.to(cfg.device)
    net.load_state_dict(torch.load(cfg.model_path))
    net.eval()
    
    mean=[0.49, 0.486, 0.482] 
    std=[0.197, 0.189, 0.187]
    val_dataset = Img_dataset(root_dir = 'dataset', is_train=False, transform=transforms.Compose([transforms.Resize((224,224)), transforms.Normalize(mean, std)]))
    val_loader = DataLoader(val_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers=cfg.num_workers)

    criterion_Loss = torch.nn.MSELoss().to(cfg.device)

    output_list = []
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            # print()
            label = data['label'].to(cfg.device)
            input = data['input'].to(cfg.device)

            output = net(input)

            op = output.cpu()

            metric_value = criterion_Loss(output, label).data.cpu()
            a = 1



    torch.cuda.empty_cache()

if __name__ == '__main__':
    cfg = parse_args()
    test_vis(cfg)
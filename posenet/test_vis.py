import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

import matplotlib.pyplot as plt

import argparse
import numpy as np
import os
import cv2
from PIL import Image
from glob import glob

from posenet_mobilenet import PoseNet

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
    parser.add_argument('--model_path', type=str, default='best_model/best_model_44.pt')

    return parser.parse_args()


# Test ------------------------------------------------


def test_vis(cfg):
    cudnn.benchmark = True

    net = PoseNet()
    net.to(cfg.device)
    net.load_state_dict(torch.load(cfg.model_path))
    net.eval()

    mean=[0.491, 0.472, 0.483] 
    std=[0.224, 0.221, 0.223]
    tf = transforms.Compose([transforms.Resize((480,270)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    data_list = glob('test_img/*.jpg')
    data_list.sort()
    print(data_list)
    for data_path in data_list:
        cv_frame = cv2.imread(data_path, cv2.IMREAD_COLOR)
        cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(cv_frame)
        frame = tf(frame).to(device)
        
        pose = net(frame.unsqueeze(0))
        # pose = pose.cpu().detach().numpy()
        print(pose)






    torch.cuda.empty_cache()

if __name__ == '__main__':
    cfg = parse_args()
    test_vis(cfg)
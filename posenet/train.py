import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms


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
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=50, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='log/OCT14_iter_pretrained_oct15_2410_4490.pth')

    return parser.parse_args()


# Train ------------------------------------------------
def train(cfg):
    writer = SummaryWriter()

    net = PoseNet()
    net.to(cfg.device)
    cudnn.benchmark = True
            
    best_model_path = 'best_model/'
    if not os.path.exists(best_model_path):
        os.mkdir(best_model_path)

    criterion_Loss = torch.nn.MSELoss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler._step_count = 0

    mean=[0.49, 0.486, 0.482] 
    std=[0.197, 0.189, 0.187]

    train_dataset = Img_dataset(root_dir = 'dataset', is_train=True, transform=transforms.Compose([transforms.Resize((224,224)), transforms.Normalize(mean, std)]))
    train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers=cfg.num_workers)

    best_metric_value = 10000
    loss_list = []
    for idx_epoch in range(st_epoch +1, cfg.n_epochs):
        loss_epoch = []
        for idx_iter, data in enumerate(train_loader):
            label = data['label'].to(cfg.device)
            input = data['input'].to(cfg.device)

            pose = net(input) # pose : (xy, wqrx) tuple

            loss = criterion_Loss(pose, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())
            torch.cuda.empty_cache()

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            # print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            writer.add_scalar('Loss / Train', float(np.array(loss_epoch).mean()), idx_epoch)

            # validation 
            metric_value = validation(net, cfg)
            print(time.ctime()[4:-5] + ' VALID_EPOCH----%5d, loss---%f' % (idx_epoch + 1, metric_value))
            writer.add_scalar('Loss / Valid', metric_value, idx_epoch)

            if metric_value < best_metric_value :
                torch.save(net.state_dict(), best_model_path + f'/best_model_{idx_epoch}.pt')
                best_metric_value = metric_value
                print('!!!BEST MODEL!!!', best_metric_value.item())


def validation(net, cfg):
    net.eval()
    criterion_Loss = torch.nn.MSELoss().to(cfg.device)

    mean=[0.49, 0.486, 0.482] 
    std=[0.197, 0.189, 0.187]
    val_dataset = Img_dataset(root_dir = 'dataset', is_train=False, transform=transforms.Compose([transforms.Resize((224,224)), transforms.Normalize(mean, std)]))
    val_loader = DataLoader(val_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers=cfg.num_workers)

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            label = data['label'].to(cfg.device)
            input = data['input'].to(cfg.device)

            output = net(input)
            
            metric_value = criterion_Loss(output, label).data.cpu()

    torch.cuda.empty_cache()

    return metric_value


if __name__ == '__main__':
    st_epoch=0

    cfg = parse_args()
    train(cfg)
    # model.load_state_dict(torch.load('./best_model/model_state_dict.pt'))
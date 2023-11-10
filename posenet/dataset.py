import torch
from torchvision import transforms, datasets
import os
from PIL import Image
import time
import pandas as pd

class Img_dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train, transform = None):
        self.root_dir = root_dir
        self.tf = transform
        self.is_train = is_train

        self.totensor = transforms.ToTensor()

        # input image
        self.img_path = root_dir + '/images'
        # label
        if self.is_train: # training phase
            self.label_path = root_dir + '/training/train.csv'
        else: # inference(test) phase
            self.label_path = root_dir + '/validation/test.csv'
        self.label_df = pd.read_csv(self.label_path)

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        label_x = self.label_df['x'].values[index]
        label_y = self.label_df['y'].values[index]
        label = torch.tensor([label_x, label_y], dtype=torch.float32).view(1, 2)

        img_name = self.label_df['img'].values[index] + '.jpg'
        
        img = Image.open(os.path.join(self.img_path, img_name))
        img = self.totensor(img)
        input_img = self.tf(img)

        data = {'input' : input_img, 'label' : label}
        # data : dict
        # input : image
        # label : tuple

        return data

if __name__ == '__main__':
    idx = 12

    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]
    train_dataset = Img_dataset(root_dir = 'dataset', is_train=True, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize(mean,std)])) # 546,804
    print(train_dataset.__getitem__(idx)['input'].shape)
    print(train_dataset.__getitem__(idx)['label'])

    train_dataset = Img_dataset(root_dir = 'dataset', is_train=False, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize(mean,std)])) # 546,804
    print(train_dataset.__getitem__(idx)['input'].shape)
    print(train_dataset.__getitem__(idx)['label'])

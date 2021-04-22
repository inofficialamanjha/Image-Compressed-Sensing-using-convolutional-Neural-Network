from os import listdir
from os.path import join
import cv2
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Grayscale
import random
import math
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.utils.data as Data
from network import CSNet_Enhanced
import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable



def calculate_valid_crop_size(crop_size, blocksize):
    return crop_size - (crop_size % blocksize)

def train_hr_transform():
    return Compose([
        Grayscale(),
        ToTensor(),
    ])

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX/math.sqrt(mse))

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, blocksize):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.hr_transform = train_hr_transform()

    def __getitem__(self, index):
        try:
            hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
            return hr_image, hr_image
        except:
            hr_image = self.hr_transform(Image.open(self.image_filenames[index+1]))
            return hr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

train_set = TrainDatasetFromFolder('location/Train_sub_images', crop_size=96, blocksize=32)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)

net = CSNet_Enhanced()

mse_loss = nn.MSELoss()

if torch.cuda.is_available():
    net.cuda()
    mse_loss.cuda()

optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[51,81], gamma=0.1)

for epoch in range(0, 100):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'g_loss': 0, }

    net.train()
    scheduler.step()

    for data, target in train_bar:
        batch_size = data.size(0)
        if batch_size <= 0:
            continue

        running_results['batch_sizes'] += batch_size

        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = net(z)
        optimizer.zero_grad()
        g_loss = mse_loss(fake_img, real_img)

        g_loss.backward()
        optimizer.step()

        running_results['g_loss'] += g_loss.item() * batch_size

        train_bar.set_description(desc='[%d] Loss_G: %.4f lr: %.7f' % (
            epoch, running_results['g_loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr']))
    
    torch.save(net.state_dict(), 'CS_net_model_large_final.pth')


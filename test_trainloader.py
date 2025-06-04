import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset, MDDataset
from net.model import AirNet

from option import options as opt
from option import opt_dict
import torchvision 

if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    subprocess.check_output(['mkdir', '-p', opt.ckpt_path])
    
    dataset_opt = opt_dict["datasets"]["train"]

    # trainset = TrainDataset(opt)
    trainset = MDDataset(dataset_opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    # Network Construction
    # net = AirNet(opt).cuda()
    # net.train()

    # Optimizer and Loss
    # optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    # CE = nn.CrossEntropyLoss().cuda()
    # l1 = nn.L1Loss().cuda()

    # Start training
    print('Start training...')
    i = 0
    for epoch in range(opt.epochs):
        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(trainloader):
            if "error" in clean_name:
                print("Error in data loading, skipping this batch.")
                continue
            
            torchvision.utils.save_image(degrad_patch_1[0], f"images/degrad_patch_1_batch_{i}.png")
            torchvision.utils.save_image(degrad_patch_2[0], f"images/degrad_patch_2_batch_{i}.png")
            torchvision.utils.save_image(clean_patch_1[0], f"images/clean_patch_1_batch_{i}.png")
            torchvision.utils.save_image(clean_patch_2[0], f"images/clean_patch_2_batch_{i}.png")
            
            i+=1
            if i > 10:
                break  
            
            

               
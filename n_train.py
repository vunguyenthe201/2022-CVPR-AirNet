import argparse
import subprocess
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset
from net.model import AirNet
from old_dataloader import DatasetFactory

from option import options as opt

if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    subprocess.check_output(['mkdir', '-p', opt.ckpt_path])
    
    # Reduce cache size and number of workers to save memory
    safe_cache_size = min(50, opt.cache_size)
    safe_workers = max(1, min(opt.num_workers, os.cpu_count()//2))
    
    # Training dataset config
    train_config = {
        "type": "high_res",
        "source_dir": opt.source_dir,
        "target_dir": opt.target_dir,
        "img_size": (opt.img_width, opt.img_height),
        "crop_size": (opt.crop_size, opt.crop_size) if opt.crop_size else None,
        "patch_size": opt.patch_size,
        "patches_per_image": opt.patches_per_image,
        "augment": opt.augment,
        "cache_size": safe_cache_size,
        "paired_naming": opt.paired_naming,
        "pair_list_file": opt.pair_list_file,
        "source_suffix": opt.source_suffix,
        "target_suffix": opt.target_suffix,
        "preload": False,  # Set to False to save memory
        "normalize": True,
        "task": opt.task,
        "random_offset": True,
        "batch_size": opt.batch_size,
        "shuffle": True,
        "num_workers": safe_workers,
        "pin_memory": False,  # Set to False to save memory
        "drop_last": True,
        "persistent_workers": False  # Set to False to save memory
    }
    
    # Validation dataset config
    val_config = None
    if opt.val_source_dir and opt.val_target_dir:
        val_config = {
            "type": "high_res",
            "source_dir": opt.val_source_dir,
            "target_dir": opt.val_target_dir,
            "img_size": (opt.img_width, opt.img_height),
            "crop_size": None,  # No cropping for validation
            "patch_size": None,  # No patches for validation
            "augment": False,   # No augmentation for validation
            "cache_size": min(20, safe_cache_size),  # Even smaller cache for validation
            "paired_naming": opt.paired_naming,
            "pair_list_file": opt.val_pair_list_file,
            "source_suffix": opt.source_suffix,
            "target_suffix": opt.target_suffix,
            "preload": False,  # Set to False to save memory
            "normalize": True,
            "task": opt.task,
            "batch_size": 1,  # Always use batch size 1 for validation
            "shuffle": False,
            "num_workers": 1,
            "pin_memory": False,
            "drop_last": False
        }
    
    

    # trainset = TrainDataset(opt)
    # trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
    #                          drop_last=True, num_workers=opt.num_workers)

    trainloader, valloader = DatasetFactory.create_data_loaders(train_config, val_config)

    # Network Construction
    net = AirNet(opt).cuda()
    net.train()

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()

    # Start training
    print('Start training...')
    for epoch in range(opt.epochs):
        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(trainloader):
            degrad_patch_1, degrad_patch_2 = degrad_patch_1.cuda(), degrad_patch_2.cuda()
            clean_patch_1, clean_patch_2 = clean_patch_1.cuda(), clean_patch_2.cuda()

            optimizer.zero_grad()

            if epoch < opt.epochs_encoder:
                _, output, target, _ = net.E(x_query=degrad_patch_1, x_key=degrad_patch_2)
                contrast_loss = CE(output, target)
                loss = contrast_loss
            else:
                restored, output, target = net(x_query=degrad_patch_1, x_key=degrad_patch_2)
                contrast_loss = CE(output, target)
                l1_loss = l1(restored, clean_patch_1)
                loss = l1_loss + 0.1 * contrast_loss

            # backward
            loss.backward()
            optimizer.step()

        if epoch < opt.epochs_encoder:
            print(
                'Epoch (%d)  Loss: contrast_loss:%0.4f\n' % (
                    epoch, contrast_loss.item(),
                ), '\r', end='')
        else:
            print(
                'Epoch (%d)  Loss: l1_loss:%0.4f contrast_loss:%0.4f\n' % (
                    epoch, l1_loss.item(), contrast_loss.item(),
                ), '\r', end='')

        GPUS = 1
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            if GPUS == 1:
                torch.save(net.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')
            else:
                torch.save(net.module.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')

        if epoch <= opt.epochs_encoder:
            lr = opt.lr * (0.1 ** (epoch // 60))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = 0.0001 * (0.5 ** ((epoch - opt.epochs_encoder) // 125))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

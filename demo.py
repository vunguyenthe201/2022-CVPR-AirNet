import argparse
import os
import subprocess

import numpy as np
import torch
import torchvision.transforms as transforms
from net.model import AirNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset_utils import TestSpecificDataset
from utils.image_io import save_image_tensor


def pad_input(input_,img_multiple_of=8):
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        return input_,height,width

def tile_eval(model,input_,tile=128,tile_overlap =32):
    b, c, h, w = input_.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h, w).type_as(input_)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
    restored = E.div_(W)

    restored = torch.clamp(restored, 0, 1)
    return restored

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--test_path', type=str, default="test/demo/", help='save path of test images')
    parser.add_argument('--output_path', type=str, default="output/demo/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
    parser.add_argument('--tile',type=bool,default=False,help="Set it to use tiling")
    parser.add_argument('--tile_size', type=int, default=128, help='Tile size (e.g 720). None means testing on the original resolution image')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    opt = parser.parse_args()

    if opt.mode == 0:
        opt.batch_size = 3
        ckpt_path = opt.ckpt_path + 'Denoise.pth'
    elif opt.mode == 1:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Derain.pth'
    elif opt.mode == 2:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Dehaze.pth'
    elif opt.mode == 3:
        opt.batch_size = 5
        ckpt_path = opt.ckpt_path + 'All.pth'

    # construct the output dir
    subprocess.check_output(['mkdir', '-p', opt.output_path])

    np.random.seed(0)
    torch.manual_seed(0)

    # Make network
    torch.cuda.set_device(opt.cuda)
    net = AirNet(opt).cuda()
    net.eval()
    net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(opt.cuda)))

    test_set = TestSpecificDataset(opt)
    testloader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    print('Start testing...')
    with torch.no_grad():
        for ([clean_name], degrad_patch, original_shape) in tqdm(testloader):
            degrad_patch = degrad_patch.cuda()
            print(original_shape)
            
            if opt.tile is False:
                restored = net(x_query=degrad_patch, x_key=degrad_patch)
                restored = transforms.Resize((original_shape[0].item(), original_shape[1].item()))(restored)
                print(restored.shape)
            else:
                print("Using Tiling")
                degrad_patch,h,w = pad_input(degrad_patch)
                restored = tile_eval(net,degrad_patch,tile = opt.tile_size,tile_overlap=opt.tile_overlap)
                restored = restored = restored[:,:,:h,:w]

            save_image_tensor(restored, os.path.join(opt.output_path, clean_name[0] + '.png'))


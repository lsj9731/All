import argparse
import os
from glob import glob

import time
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from models import TSN
from transforms import *
from dataset_mod import TSNDataSet
from ops import ConsensusModule
from tqdm import tqdm

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, choices=['Flow', 'RGB'], default='Flow')
    parser.add_argument("--checkpoint", type=str, default='./weights/UCF101_Flow__flow_model_best.pth.tar')
    # filelist path
    parser.add_argument("--vid_lists_path", default="/root/server_repository/data", type=str)
    parser.add_argument("--tsn_save_path", default="/root/all_dir/tsn_outputs/flow", type=str)
    parser.add_argument("--use_gpu_num", default=0, type=int)
    parser.add_argument("--num_segments", default=25, type=int)
    return parser.parse_args()

def _generate_input_list():
    samples_lst = glob(os.path.join(args.frames_path, args.modality.lower(), '*'))
    with open(os.path.join(args.vid_lists_path, 'filelist.txt'), 'w') as f:
        for sample in samples_lst:
            num_frames = len(glob(os.path.join(sample, '*.jpg'))) // 2
            print('num_frames : ', num_frames)
            line = sample + ' ' + str(num_frames) + '\n'
            print('line:',line)
            f.write(line)
            
def _save_csv(tensor, name):
    cols = [f"f{i}" for i in range(tensor.shape[1])]
    out_df = pd.DataFrame(tensor, columns=cols)
    out_df.to_csv(os.path.join(args.tsn_save_path, f"v__{name}.csv"), index=False)
    folder_m = 'spatial' if args.modality == 'RGB' else 'temporal'
    print(f"Saved to - ", os.path.join(args.tsn_save_path, folder_m, f"v__{name}.csv"))
            
if __name__ == "__main__":
    args = _parse_args()
    #_generate_input_list()
    torch.cuda.empty_cache()
    device = torch.device(f'cuda:{args.use_gpu_num}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    print('modality : ', args.modality)

    # Load Model
    net = TSN(101, 1, args.modality,
          base_model='BNInception',
          consensus_type='avg',
          dropout=0.7)
    checkpoint = torch.load(args.checkpoint)
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)
    net.to(device)
    net.eval()
    # Prepare Dataset
    transform = torchvision.transforms.Compose([
                       #GroupOverSample(net.input_size, net.scale_size),
                       GroupResize(net.input_size),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])
    dataset = TSNDataSet('', os.path.join(args.vid_lists_path, 'filelist.txt'), 
                         modality=args.modality, 
                         image_tmpl='flow_'+"{}_{:05d}.jpg" if args.modality == 'Flow' else 'img_{:05d}.jpg',
                         transform=transform,
                         new_length=1 if args.modality=='RGB' else 5,
                         num_segments=args.num_segments,
                         test_mode=True
                         )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    print(f"Processing {len(dataset)} videos.")
    # Forward Propagate
    length = 3 if args.modality == 'RGB' else 10
    if args.modality == 'RGB':
        with torch.no_grad():
            for inp_tensor, sample_name, sample_len in dataloader:
                start = time.time()
                one_total_data = []
                inp_tensor = inp_tensor.view(-1, length, inp_tensor.size(2), inp_tensor.size(3))
                batch_size = 32
                epoch = inp_tensor.size()[0] // batch_size
                for i in tqdm(range(epoch)):
                    time.sleep(1/epoch)
                    if i == epoch-1:
                        sample_tensor = inp_tensor[batch_size*i:].to(device)
                    else:
                        sample_tensor = inp_tensor[batch_size*i:batch_size*(i+1)].to(device)
                    out = net(sample_tensor)
                    out_npy = out.cpu().numpy()
                    one_total_data.append(out_npy)
                print(' time : ', time.time() - start, '\n\n')
                one_total_data_npy = np.concatenate(one_total_data)
                one_total_data_npy = np.squeeze(one_total_data_npy)
                _save_csv(one_total_data_npy, sample_name[0])
                torch.cuda.empty_cache()
        print("Finished TSN RGB Feature Extraction.")
    elif args.modality == 'Flow':
        with torch.no_grad():
            for inp_tensor, sample_name, sample_len in dataloader:
                start = time.time()
                one_total_data = []
                inp_tensor = inp_tensor.view(-1, length, inp_tensor.size(2), inp_tensor.size(3))
                for i in tqdm(range(inp_tensor.size()[0])):
                    time.sleep(1/inp_tensor.size()[0])
                    sample_tensor = inp_tensor[i].to(device)
                    out = net(sample_tensor)
                    out_npy = out.detach().cpu().numpy()
                    out_npy = out_npy.reshape(1, -1)
                    one_total_data.append(out_npy)
                    del out
                    del out_npy
                print(' time  : ', time.time() - start, '\n\n')
                one_total_data_npy = np.concatenate((one_total_data), axis=0)
                print('concatenate data array : ', one_total_data_npy)
                one_total_data_npy = np.squeeze(one_total_data_npy)
                print('save total data array : ', one_total_data_npy)
                _save_csv(one_total_data_npy, sample_name[0])
                torch.cuda.empty_cache()
        print("Finished TSN Flow Feature Extraction.")
    

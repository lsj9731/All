import os
import argparse
# import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
from src.i3dpt import I3D
from PIL import Image
import time

class VideoDataset(Dataset):
    def __init__(self, args):
        self.data = glob(os.path.join(args.src_folder, '*.'+args.ext))
        self.args  = args

    def __getitem__(self, idx):
        vid_path = self.data[idx]
        if not vid_path.endswith('.npy'):
            vid_np = video_to_numpy(vid_path, self.args)
        else:
            vid_np = np.load(vid_path)
            print('vid_np shape : ', vid_np.shape)
            vid_np = np.squeeze(vid_np)
            print('return data : ', vid_np.reshape(1, vid_np.shape[-1], vid_np.shape[0], 224, 224).shape)
            vid_len = vid_np.shape[0]
        return vid_np.reshape(vid_np.shape[-1], vid_np.shape[0], 224, 224), vid_path, vid_len
    
    def __len__(self):
        return len(self.data)

def video_to_numpy(path, args):
    frames = []
    cap = cv2.VideoCapture(path)
    ret = True    
    frame_count = 0
    while ret:
        if frame_count == args.max_length: # max frame length = 3000 / only use first 3000 frames for now.
            break
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (224, 224))
            frames.append(img)
        frame_count += 1
    video = np.stack(frames, axis=0) # (T, H, W, C)
    return video

def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def _load_i3d_model(path):
    if args.modality == 'rgb':
        i3d = I3D(num_classes=400) # num classes = pretrained num classes (=400)
        i3d.load_state_dict(torch.load('./model/model_rgb.pth')) # load RGB Pretrained
    elif args.modality == 'flow':
        i3d = I3D(num_classes=400, modality='flow') # num classes = pretrained num classes (=400)
        i3d.load_state_dict(torch.load('./model/model_flow.pth')) # load flow pretrained
    i3d.to(device)
    i3d.eval()
    return i3d

def extract_feature(inp, model):
    with torch.no_grad():
        out = model.forward(inp.to(device).float(), early_stop='avg') # STOP at AvgPool
    out = out.view(out.shape[2], -1)
    return out

def main(args):
    data_dir = args.src_folder
    read_dir = os.listdir(data_dir)
    i3d = _load_i3d_model(args.weight_path)
    if args.modality == 'rgb':
        start = time.time()
        for folders in read_dir:
            # 전체 데이터 + 비디오 경로
            data_dirs = os.path.join(data_dir, folders)
            datas = os.listdir(data_dirs)
            rgb_all = []
            print('load Image ...')
            for i, folder_name in enumerate(datas):
                img_list_dir = os.path.join(data_dirs, folder_name)
                img_list = os.listdir(img_list_dir)
                [rgb_all.append(Image.open(os.path.join(img_list_dir, img)).convert('RGB')) for img in img_list if 'img' in img]
            print('Done.\n')

            out_images_rgb = list()
            print('preprocessing the Image ...')
            [out_images_rgb.append(np.array(rgb_image.resize((224, 224))) / 255.) for rgb_image in rgb_all]
            print('Done.\n')

            # del image
            del rgb_all

            print('Concatenate npy array ...')
            out_images_rgb = np.concatenate(out_images_rgb).reshape(1, -1, 224, 224, 3)
            print('Done.\n')

            print('before Extraction data shape : ', out_images_rgb.shape, '\n')
            vid_name = data_dirs.split('/')[-1]
            print(vid_name)
            vid_len = out_images_rgb.shape[1]

            # before Extraction data shape :  (1, 3, 147465, 224, 224)
            out_images_rgb = out_images_rgb.reshape(1, 3, -1, 224, 224)
            data_size = out_images_rgb.shape[2]

            extract_features = []
            clip_len = 3000
            batch_size = data_size // clip_len
            print('Extraction .. ')
            for i in tqdm(range(batch_size)):
                time.sleep(1/batch_size)
                if i == batch_size:
                    sample_data = out_images_rgb[:, :, clip_len*i:, :, :]
                else:
                    sample_data = out_images_rgb[:, :, clip_len*i:clip_len*(i+1), :, :]
                sample_data = torch.Tensor(sample_data).to(device)
                print('data size:', sample_data.size())
                out = extract_feature(sample_data, i3d)
                out = out.cpu().detach().numpy()
                extract_features.append(out)
                del out, sample_data
            del out_images_rgb
            print('Done.\n')
            list_to_npy = np.array(extract_features)
            del extract_features
            print('extract_features : ', list_to_npy.shape)
            extract_features_npy = np.concatenate(list_to_npy)
            del list_to_npy
            print('concatenate data : ', extract_features_npy.shape)
                
            print('Save npy ...')
            np.save(args.out_folder+'/rgb_'+vid_name+'.npy', extract_features_npy)
            del extract_features_npy
            print('Done.\n')
            print('end time : ', time.time() - start)
            torch.cuda.empty_cache()
    elif args.modality == 'flow':
        start = time.time()
        for folders in read_dir:
            # 전체 데이터 + 비디오 경로
            data_dirs = os.path.join(data_dir, folders)
            datas = os.listdir(data_dirs)
            flow_x_all = []
            flow_y_all = []
            print('load Image ...')
            for i, folder_name in enumerate(datas):
                img_list_dir = os.path.join(data_dirs, folder_name)
                img_list = os.listdir(img_list_dir)
                [flow_x_all.append(Image.open(os.path.join(img_list_dir, img)).convert('L')) for img in img_list if 'flow_x' in img]
                [flow_y_all.append(Image.open(os.path.join(img_list_dir, img)).convert('L')) for img in img_list if 'flow_y' in img]
            print('Done.\n')

            out_images_flow_X = list()
            out_images_flow_y = list()
            print('preprocessing the Image ...')
            [out_images_flow_X.append(np.array(flow_x_image.resize((224, 224))) / 255.) for flow_x_image in flow_x_all]
            [out_images_flow_y.append(np.array(flow_y_image.resize((224, 224))) / 255.) for flow_y_image in flow_y_all]
            print('Done.\n')
            
            del flow_x_all, flow_y_all

            print('Concatenate npy array ...')
            out_images_flow_X = np.concatenate(out_images_flow_X).reshape(1, -1, 224, 224, 1)
            out_images_flow_y = np.concatenate(out_images_flow_y).reshape(1, -1, 224, 224, 1)
            concat_flow = np.concatenate((out_images_flow_X, out_images_flow_y), axis = -1)
            print('Done.\n')

            del out_images_flow_X, out_images_flow_y

            print('before Extraction data shape : ', concat_flow.shape, '\n')
            vid_name = data_dirs.split('/')[-1]
            vid_len = concat_flow.shape[1]

            # before Extraction data shape :  (1, 2, 147465, 224, 224)
            concat_flow = concat_flow.reshape(1, 2, -1, 224, 224)
            data_size = concat_flow.shape[2]

            extract_features = []
            clip_len = 3000
            batch_size = data_size // clip_len
            print('Extraction .. ')
            for i in tqdm(range(batch_size)):
                time.sleep(1/batch_size)
                if i == batch_size:
                    sample_data = concat_flow[:, :, clip_len*i:, :, :]
                else:
                    sample_data = concat_flow[:, :, clip_len*i:clip_len*(i+1)+3, :, :]
                sample_data = torch.Tensor(sample_data).to(device)
                out = extract_feature(sample_data, i3d)
                out = out.cpu().detach().numpy()
                extract_features.append(out)
                del out, sample_data
            del concat_flow
            print('Done.\n')

            list_to_npy = np.array(extract_features)
            del extract_features
            
            print('extract_features : ', list_to_npy.shape)
            extract_features_npy = np.concatenate(list_to_npy)
            del list_to_npy
            
            print('concatenate data : ', extract_features_npy.shape)
            
            print('Save npy ...')
            np.save(args.out_folder+'/flow_'+vid_name+'.npy', extract_features_npy)
            print('Done.\n')
            del extract_features_npy
            print('end time : ', time.time() - start)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract I3D features using Kinetics400 pretrained weight.")
    # 데이터 셋 위치
    parser.add_argument('--src_folder', type=str, default='/root/server_repository/data')
    parser.add_argument('--out_folder', type=str, default='/root/data_output/flow')
    parser.add_argument('--modality', choices=['rgb', 'flow'], default='flow')
    parser.add_argument('--ext', type=str, choices=['mp4', 'wmv', 'avi', 'npy'], default='npy')
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--weight_path', type=str, default='./model/model_flow.pth')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=3000)
    args = parser.parse_args()

    if args.modality == 'flow' and args.ext != 'npy':
        raise AssertionError("Flow 에서 I3D 추출시, 현재 .npy 확장자만 변환 가능")
    # GPU SETTING
    device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(f"Pytorch I3D extraction running on: {device}")
    
    if not os.path.exists(args.out_folder):
        os.mkdir(args.out_folder)
    # Run
    main(args)
    print("Done")

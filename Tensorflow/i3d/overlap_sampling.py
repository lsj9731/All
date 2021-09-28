import numpy as np
import argparse
import tensorflow as tf
import os
import i3d
from tqdm import tqdm
import time

_IMAGE_SIZE = 224
NUM_CLASSES = 400

os.environ["CUDA_VISIBLE_DEVICES"]='0'

def get_RGB_feature(eval_type, data_, check_point, end_point, video_id, save_path):
    tf.reset_default_graph()
    frame = data_.shape[1]
    if frame <= 8000:
        _batch_size = 2
        overlap_ratio_a = 0
        overlap_ratio_b = 0
    else:
        _batch_size = frame // 4000
        overlap_ratio_a = 0
        overlap_ratio_b = 0
    
    batchs = frame // _batch_size
    overlap_size_after = int(batchs * overlap_ratio_a)
    overlap_size_befor = int(batchs * overlap_ratio_b)
    rgb_model = i3d.InceptionI3d(NUM_CLASSES, spatial_squeeze=True, final_endpoint=end_point)

    all_data = []
    for i in range(_batch_size):
        if i == _batch_size-1:
            tf.reset_default_graph()
            batch_data = data_[0, i*(batchs - overlap_size_after):]
            bat_size = batch_data.shape[0]
            batch_data = batch_data.reshape(1, bat_size, _IMAGE_SIZE, _IMAGE_SIZE, 3)
            rgb_input = tf.placeholder(tf.float32, shape=(1, bat_size, _IMAGE_SIZE, _IMAGE_SIZE, 3))
        else:
            tf.reset_default_graph()
            batch_data = data_[0, i*(batchs - overlap_size_after):((i+1)*(batchs)) + overlap_size_befor]
            bat_size = batch_data.shape[0]
            batch_data = batch_data.reshape(1, bat_size, _IMAGE_SIZE, _IMAGE_SIZE, 3)
            rgb_input = tf.placeholder(tf.float32, shape=(1, bat_size, _IMAGE_SIZE, _IMAGE_SIZE, 3))
        with tf.variable_scope('RGB'):
            rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
            rgb_variable_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'RGB':
                    if eval_type == 'rgb':
                        rgb_variable_map[variable.name.replace(':0', '')] = variable

            rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        if eval_type == 'rgb':
            model_logits = rgb_logits
        else:
            model_logits = rgb_logits + flow_logits
        model_predictions = tf.nn.softmax(model_logits)

        with tf.Session() as sess:
            feed_dict = {}
            if eval_type == 'rgb':
                rgb_saver.restore(sess, check_point[eval_type])
            feed_dict[rgb_input] = batch_data
            out_logits, _ = sess.run([model_logits, model_predictions], feed_dict=feed_dict)
            out_logits = np.array(out_logits)
            out_logits = np.squeeze(out_logits)
            all_data.append(out_logits)
            
    all_data = np.concatenate(all_data)
    np.save(save_path+video_id, all_data)


def get_FLOW_feature(eval_type, data_, check_point, end_point, video_id, save_path):
    tf.reset_default_graph()
    frame = data_.shape[1]
    if frame <= 8000:
        _batch_size = 2
        overlap_ratio_a = 0
        overlap_ratio_b = 0.001
    else:
        _batch_size = frame // 6000
        overlap_ratio_a = 0
        overlap_ratio_b = 0.001

    batchs = frame // _batch_size
    overlap_size_after = int(batchs * overlap_ratio_a)
    overlap_size_befor = int(batchs * overlap_ratio_b)
    flow_model = i3d.InceptionI3d(NUM_CLASSES, spatial_squeeze=True, final_endpoint=end_point)

    all_data = []
    for i in range(_batch_size):
        if i == _batch_size-1:
            tf.reset_default_graph()
            batch_data = data_[0, i*(batchs - overlap_size_after):]
            bat_size = batch_data.shape[0]
            batch_data = batch_data.reshape(1, bat_size, _IMAGE_SIZE, _IMAGE_SIZE, 2)
            flow_input = tf.placeholder(tf.float32, shape=(1, bat_size, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        else:
            tf.reset_default_graph()
            batch_data = data_[0, i*(batchs - overlap_size_after):((i+1)*(batchs)) + overlap_size_befor]
            bat_size = batch_data.shape[0]
            batch_data = batch_data.reshape(1, bat_size, _IMAGE_SIZE, _IMAGE_SIZE, 2)
            flow_input = tf.placeholder(tf.float32, shape=(1, bat_size, _IMAGE_SIZE, _IMAGE_SIZE, 2))
     
        with tf.variable_scope('Flow'):
            flow_logits, _ = flow_model(flow_input, is_training=False, dropout_keep_prob=1.0)
            flow_variable_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'Flow':
                    flow_variable_map[variable.name.replace(':0', '')] = variable
            flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

        if eval_type == 'flow':
            model_logits = flow_logits
        else:
            model_logits = rgb_logits + flow_logits
        model_predictions = tf.nn.softmax(model_logits)

        with tf.Session() as sess:
            feed_dict = {}
            if eval_type == 'flow':
                flow_saver.restore(sess, check_point[eval_type])
            feed_dict[flow_input] = batch_data
            out_logits, _ = sess.run([model_logits, model_predictions], feed_dict=feed_dict)
            out_logits = np.array(out_logits)
            out_logits = np.squeeze(out_logits)
            all_data.append(out_logits)
            
    all_data = np.concatenate(all_data)
    np.save(save_path+video_id, all_data)                



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='rgb')
    parser.add_argument('--d_type', type=str, default='val')
    parser.add_argument('--end_point', type=str, default='Mixed_5c')

    args = parser.parse_args()

    if args.type not in ['rgb', 'flow'] :
        raise ValueError('Bad `--type`, must be one of rgb, flow')

    if args.type == 'rgb' and args.d_type == 'val':
        data_dir = ''
        save_dirs = ''
    elif args.type == 'rgb' and args.d_type == 'test':
        data_dir = ''
        save_dirs = ''
    elif args.type == 'flow' and args.d_type == 'val':
        data_dir = ''
        save_dirs = ''
    elif args.type == 'flow' and args.d_type == 'test':
        data_dir = ''
        save_dirs = ''

    if args.type == 'rgb':
        print('data_dir : ', data_dir)
        print('save_dir : ', save_dirs)
        _CHECKPOINT_PATHS = {'rgb': 'data/checkpoints/rgb_scratch/model.ckpt'}
        end_point = args.end_point
        video_list = os.listdir(data_dir)
        if 'flow' in video_list:
            flow_index = video_list.index('flow')
            video_list.pop(flow_index)
        save_dir = os.listdir(save_dirs)

        for save_file in save_dir:
            _file = save_file
            if _file in video_list:
                file_index = video_list.index(_file)
                video_list.pop(file_index)

        for i in tqdm(video_list):
            print('video_name : ', i)
            time.sleep(1/len(video_list))
            samples = np.load(data_dir+i)
            print('sample shape : ', samples.shape)
            _SAMPLE = samples
            get_RGB_feature(args.type, _SAMPLE, _CHECKPOINT_PATHS, end_point, i, save_dirs)
        print('All Done.')
    elif args.type == 'flow':
        print('data_dir : ', data_dir)
        print('save_dir : ', save_dirs)
        _CHECKPOINT_PATHS = {'flow': 'data/checkpoints/flow_scratch/model.ckpt'}
        end_point = args.end_point
        video_list = os.listdir(data_dir)
        save_dir = os.listdir(save_dirs)

        for save_file in save_dir:
            _file = save_file[:-4]
            _file = _file+'w.npy'
            print('file name : ', _file)
            if _file in video_list:
                file_index = video_list.index(_file)
                video_list.pop(file_index)

        for i in tqdm(video_list):
            print('video_name : ', i)
            time.sleep(1/len(video_list))
            video_name = i[:-5]
            samples = np.load(data_dir+i)
            print('sample shape : ', samples.shape)
            _SAMPLE = samples
            get_FLOW_feature(args.type, _SAMPLE, _CHECKPOINT_PATHS, end_point, video_name, save_dirs)
        print('All Done.')

if __name__ == '__main__':
  main()
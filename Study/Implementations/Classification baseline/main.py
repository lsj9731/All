import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os
from opts import parser
from transforms import *
from model import *
from dataset import *
from tensorflow import keras
from tqdm import tqdm
import tensorflow_hub as hub
import datetime
import tensorboard

IMG_SIZE = 224

def main():
    args = parser.parse_args()
    best_loss = 100

    if args.dataset == 'cifar10':
        num_class = 10
    elif args.dataset == 'fashion-mnist':
        num_class = 10
    elif args.dataset == 'mnist':
        num_class = 10
    elif args.dataset == 'cifar100':
        num_class = 100
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    # get train, valid dataset
    dataset_dir = args.dirs
    
    # Construct model
    print('Construct model')
    model = Counstructor(num_class, drop_rate=args.dropout, model_version = args.version)
    print('Done.')

    # load model checkpoint
    if args.checkpoint:
        print('Get checkpoint {}'.format(args.checkpoint))
        loaded = tf.saved_model.load(args.checkpoint)
        model = build_model(loaded)

    # Preprocessing options
    image_transforms = []

    # define DataLoader
    train_datalist = Dataloader(dataset_dir, args.batch_size, image_transforms, mode = 'train')
    valid_datalist = Dataloader(dataset_dir, args.batch_size, image_transforms, mode = 'valid')

    # define loss function & optimizer & model compile
    optim = tf.keras.optimizers.SGD(learning_rate=args.lr, decay=0.001, momentum=0.9, nesterov=True)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # tensorboard --logdir=./logs/
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    # model training
    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        print('Model training ...')

        for images, labels in tqdm(train_datalist):
            time.sleep(1 / len(train_datalist))
            train_l = train_step(model, loss_object, optim, images, labels, train_loss, train_accuracy)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        print('Model Evaluation ...')

        for test_images, test_labels in tqdm(valid_datalist):
            time.sleep(1 / len(valid_datalist))
            test_l = test_step(model, loss_object, test_images, test_labels, test_loss, test_accuracy)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)


        print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result()}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result()}')

        if epoch % 10 == 0:
            print('Create Model Checkpoint on {} epochs'.format(epoch))
            tf.saved_model.save(model, os.path.join(args.resume, 'Model_Checkpoint_{}_epochs'.format(epoch)))


        if test_loss.result() < best_loss:
            best_loss = test_loss.result()
            print('Create Model Checkpoint on {} epochs'.format(epoch))
            tf.saved_model.save(model, os.path.join(args.best, 'Model_Best_Checkpoint_{}_epochs'.format(epoch)))


def train_step(model, loss_object, optimizer, images, labels, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    return loss

def test_step(model, loss_object, images, labels, test_loss, test_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

    return t_loss

def build_model(loaded):
    x = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_x')
    # KerasLayer로 감싸기
    keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
    model = tf.keras.Model(x, keras_layer)
    return model

if __name__ == '__main__':
    print('Tensorflow version : ', tf.__version__)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 텐서플로가 첫 번째 GPU만 사용하도록 제한
        try:
            print('start with GPU 0 & 1')
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)

    main()
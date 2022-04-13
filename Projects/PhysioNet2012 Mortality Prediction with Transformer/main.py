import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import scipy
import os
import dataloader
import time
import model
from opts import parser
from tqdm import tqdm
import tensorflow_hub as hub
import datetime
import tensorboard
import custom_callbacks
from sklearn.utils.class_weight import compute_sample_weight


def main():
    args = parser.parse_args()
    best_loss = 100

    # PhysioNet2012 goal is binary classification
    num_class = 1
    train_dataset = dataloader.Dataloader(args.dirs, args.batch, 'Training', 'PhysioNet2012')
    valid_dataset = dataloader.Dataloader(args.dirs, args.batch, 'Validation', 'PhysioNet2012')

    # output_activation, output_dims, n_dims, n_heads, n_layers, dropout, attn_dropout, aggregation_fn, max_timescale):
    print('Construct model')
    Transformer = model.TransformerModel('sigmoid', num_class, args.dims, args.heads, args.layers, args.dropout, args.dropout, 'mean', 100.0)
    print('Done.')

    if args.checkpoint is not None:
        print('Get saved model weights !')
        print('We will resume model training.')
        Transformer.load_weights(args.checkpoint)

    optim = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # tensorboard --logdir=./logs/
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    reduce_rl_plateau = custom_callbacks.ReduceLROnPlateau(patience=30,
                              factor=0.0001,
                              verbose=1, 
                              optim_lr=optim.learning_rate, 
                              reduce_lin=True)

    reduce_rl_plateau.on_train_begin()
    epochs_no_improve, n_epochs_stop = 0, args.early_stop
    
    for epoch in tqdm(range(args.epochs)):
        # Reset the metrics at the start of the next epoch
        time.sleep(1 / args.epochs)

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        
        for datas, labels in train_dataset:
            train_step(Transformer, loss_object, optim, datas, labels, train_loss, train_accuracy)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            reduce_rl_plateau.on_batch_end()

            print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result()}, '
            f'Learning rate: {optim.learning_rate}, ')

        for test_datas, test_labels in valid_dataset:
            test_step(Transformer, loss_object, test_datas, test_labels, test_loss, test_accuracy)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
            
            print(
            f'Epoch {epoch + 1}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result()}')

        if epoch % 10 == 0:
            print('Create Model Checkpoint on {} epochs'.format(epoch))
            Transformer.save_weights(os.path.join(args.resume+'/{}epochs/'.format(epoch), 'Model_Checkpoint_{}_epochs'.format(epoch)))

        if test_loss.result() < best_loss:
            best_loss = test_loss.result()
            best_model = Transformer
            best_epoch = epoch
            print('Create Model Checkpoint on {} epochs'.format(epoch))
            Transformer.save_weights(os.path.join(args.best+'/{}epochs/'.format(epoch), 'Model_Best_Checkpoint_{}_epochs'.format(epoch)))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!' )
            early_stop = True
            break

    if early_stop:
        print("Stopped and best weights models save")
        best_model.save_weights(os.path.join(args.earlystop+'/{}epochs/'.format(best_epoch), 'Model_Best_Checkpoint_{}_epochs'.format(best_epoch)))
        

def train_step(model, loss_object, optimizer, datas, labels, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(datas, training=True)
        # loss = loss_object(labels, predictions)
        get_class_weights = _class_weights(labels)
        loss = loss_object(labels, predictions, sample_weight=get_class_weights)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

def test_step(model, loss_object, datas, labels, test_loss, test_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(datas, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

def _class_weights(labels):
    cw = compute_sample_weight(class_weight = "balanced" , y = labels)

    return cw

if __name__ == "__main__":
    main()
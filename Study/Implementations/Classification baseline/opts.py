import argparse
parser = argparse.ArgumentParser(description="Tensorflow2 study args parser")
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'mnist', 'fashion-mnist'])

# ========================= Dataset Configs ==========================
parser.add_argument('--dirs', default='C:/Users/82107/Desktop/Python/Data/cifar-100-python/', type=str, metavar='Dir',
                    help='Dataset Dir')
parser.add_argument('--aug', default=False, type=bool, metavar='Aug',
                    help='Data Augmentation options')

# ========================= Model Configs ==========================
parser.add_argument('--version', default='VGG16', type=str,
                    metavar='V', help='Select to Model Version (default : VGG16)')
parser.add_argument('--dropout', default=0.8, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=50, type=float, metavar='LRSteps', 
                    help='epochs to decay learning rate by N interval')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
                

# ========================= Monitor Configs ==========================
parser.add_argument('--p_eval', default=10, type=int,
                    metavar='Evalute', help='print evaluate')

# ========================= Runtime Configs ==========================
parser.add_argument('--gpus', nargs='+', type=int, default=[0])
parser.add_argument('--worker', type=int, default=0, help='num cpu worker')
parser.add_argument('--resume', default='./check_point', type=str, metavar='PATH',
                    help='save to model parameters (default: none)')
parser.add_argument('--best', default='./best_weights', type=str, metavar='PATH',
                    help='path to best weights (default: none)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
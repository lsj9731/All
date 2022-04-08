import argparse

parser = argparse.ArgumentParser(description="Transformer for physionet2012 modality prediction")

# ========================= Dataset Configs ==========================
parser.add_argument('--dirs', default='C:/Users/82107/Desktop/Python/Data/Transformer_Physionet2012/after_preprocessing', type=str, metavar='Dir',
                    help='Dataset Dir')

# ========================= Model Configs ============================
parser.add_argument('--dropout', default=0.0, type=float,
                    metavar='DO', help='dropout ratio (default: 0.0)')
parser.add_argument('--dims', default=128, type=int,
                    metavar='DO', help='Transformer model dimension (default: 128)')
parser.add_argument('--layers', default=1, type=int,
                    metavar='DO', help='num of transformer block (default: 1)')
parser.add_argument('--heads', default=4, type=int,
                    metavar='DO', help='num of MultiAttention (default: 4)')

# ========================= Learning Configs =========================
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.000001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

# ========================= Monitor Configs ==========================
parser.add_argument('--early_stop', default=30, type=int,
                    metavar='Evalute', help='print evaluate')

# ========================= Runtime Configs ==========================
parser.add_argument('--resume', default='./check_point', type=str, metavar='PATH',
                    help='save to model parameters (default: none)')
parser.add_argument('--best', default='./best_weights', type=str, metavar='PATH',
                    help='path to best weights (default: none)')
parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--earlystop', default='./early_stop', type=str, metavar='PATH',
                    help='path to early stopping checkpoint (default: none)')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time
from tqdm import tqdm
import argparse
from terminaltables import *

pred = pickle.load(open('./data/pred_dump.pc', "rb"))

print('pred : ', pred)
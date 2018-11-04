import pandas as pd
import numpy as np
from tqdm import tqdm

import random
import math
import time
import os
import six
import chainer
import chainer.functions as F
import chainer.links as L
import zipfile
from chainer.links import caffe
from chainer import link, Chain, optimizers, Variable, initializers, training
from PIL import Image
import pickle
import gzip
import h5py
#import leveldb
from scipy import ndimage, misc
try:
    import cupy
except:
    xp=np

import logging
from datetime import datetime
import argparse
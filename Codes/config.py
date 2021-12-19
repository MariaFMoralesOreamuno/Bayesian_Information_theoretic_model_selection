"""
Author: María Fernanda Morales Oreamuno
Date created: 28/05/2021
Last update: 30/07/2021

File contains the modules and folders to import for all codes to run.
"""

try:
    import sys
    import os
    import pickle
    import glob
    import time
    import logging
    import math
    import itertools
    import warnings
except ModuleNotFoundError as e:
    print('ModuleNotFoundError: Missing basic libraries (required: sys, os, pickle, glob, time, logging, math,'
          ' itertools')
    print(e)

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter
    import seaborn as sn
    import scipy.stats as stats
    from scipy.io import loadmat
    from tqdm import tqdm
    from tqdm import trange
except ModuleNotFoundError as e:
    print('ModuleNotFoundError: Missing additional libraries (required: numpy, pandas, matplotlib, seaborn,'
          ' scipy.stats, tqdm')
    print(e)




#
import os
import numpy as np
#import matplotlib as mpl
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision.datasets import ImageFolder   
from torch.utils.data import DataLoader  
import torchvision.transforms as transforms
import torchvision
import glob
import random


def get_label(path):
	
	father_doc=os.path.dirname(path)
	father_base=os.path.basename(father_doc)
	return father_base
	

def get_list()
	path=r'D:\脑科学\NYU\training_data'
	pic=glob.glob(r'{}\*\snwmrda*_session_1_rest_1_aal*.1D'.format(path))
	print(pic[0])
	print("数据长度:",len(pic))
	random.shuffle(pic)
	print(pic[:3])
	label=[]
	for i in range(len(pic)):
		label.append(get_label(pic[i]))
	print("获得label:",len(label)," 个")
	return pic,label




















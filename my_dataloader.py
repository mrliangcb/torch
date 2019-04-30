#读图队列
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


#从子路径获取label
def get_label(path):
	
	father_doc=os.path.dirname(path)
	father_base=os.path.basename(father_doc)
	return int(father_base)

def read_pic(path):
	nparray=np.array([[1,2,3],[4,5,6],[7,8,9]])
	return nparray
	
	
#函数型迭代器  传入数据文件夹，装着0,1
def get_data(doc,batch,epoch):
	# pic=glob.glob(r'{}/*/snwmrda*_session_*_rest_*_aal*.1D'.format(doc)) #根据后缀搜索文件
	#假设获得为
	pic=['./0/1','./1/2','./0/3']
	
	epoch_index=0
	while epoch_index<epoch:#0,1
		############  拼 全名,label列表  #############
		random.shuffle(pic) #一个epoch的开始就打乱顺序
		list_all=[]
		for i in range(len(pic)):# 按照图片去找类别
			list_all.append([pic[i],get_label(pic[i])])  #[ [图全名，label],[  ]]   一个epoch的pic和lab打包好了
		##############################################
	
		######### 遍历一个epoch，满一个batch输出 ##################
		i=0
		batch_out=[]
		batch_index=0
		while i <len(list_all):
			batch_out.append([read_pic(list_all[i][0]),list_all[i][1]])#[[矩阵,0],[]]
			i+=1
			batch_index+=1
			if batch_index>=batch:
				yield batch_out #当打包batch个的时候输出
				#del batch_out[:]#清零
				batch_out=[]
				batch_index=0
		###########################################################
		
		epoch_index+=1
		
b=get_data(1,3,3)

for i in b:
	print(i)
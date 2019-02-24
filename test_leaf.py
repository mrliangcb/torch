import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


data_ar=np.ones(shape=(2,2),dtype='float32')
x_tensor = torch.Tensor(data_ar)
x = Variable(x_tensor, requires_grad=True)

data_ar2=np.ones(shape=(2,2),dtype='float32')
x_tensor2 = torch.Tensor(data_ar2)
x2 = Variable(x_tensor2, requires_grad=True)

def test(a,b):
	flat = torch.Tensor(1,2).zero_()
	flat[0,0]=a[0,0]#可见tensor是可以修改的
	cal=torch.Tensor(2).zero_()
	for i in range(2):
		cal[i]=a[i,0]+b[i,0]
	flat[0,1]=cal.sum()
		#print("迹为多少:",S_tr)#27(3*9)和54(3*18)
	#等号两边是不可以有同样的tensor的，克隆的不算同一个人
	flat2=flat.clone()#新建一个变量，这样等号的左右两边就不会有同一个人了
	flat[0,1]=torch.div(flat2[0,1],2)#这整个过程中S2因为是克隆出来的，所以一直都是非叶子变量#这个操作是不可以的
	
	return flat

y=test(x,x2)
y2=y.sum()
y2.backward()








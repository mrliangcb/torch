#关于softmax


def soft(list):
	result0=math.pow(math.e,list[0])/(math.pow(math.e,list[0])+math.pow(math.e,list[1]))
	result1=math.pow(math.e,list[1])/(math.pow(math.e,list[0])+math.pow(math.e,list[1]))
	return result0,result1
	
list=[-0.7280,-0.6594]
print(soft(list))


import torch.nn.functional as F
import torch 
import numpy as np
x=torch.Tensor(np.array([list]))
print(x)
x1 = F.log_softmax(x, dim=1)
print(x1)
x2 = F.softmax(x, dim=1)
print(x2)

#
import torch
import numpy as np

#一维
a=torch.ones(3)
b=torch.Tensor(np.array([1,2,3]))
print(a)
print(b)
c=torch.dot(a,b)#一维乘以一维
print(c)


#二维 横乘以竖
a=torch.ones(3,3)
b=torch.Tensor(np.array([[1,2,3],[4,5,6],[7,8,9]]))
d=torch.mm(a,b)
print(d)


#二维 对应点相乘
a=torch.Tensor(np.array([[1,2],[4,5]]))
aa=torch.ones(2,116)
b=torch.Tensor(np.array([[1,2],[4,5]]))
e=torch.mul(a,b)
print(e)



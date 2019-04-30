



import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
conv=nn.Conv2d(
				in_channels=1,
				out_channels=16,
				kernel_size=5,
				stride=1,
				padding=2
				)
conv2=nn.Linear(32*7*7, 10)
# print(nn.parameters)#没有parameters
print(conv.weight)
print(conv2.weight)
print('大小',conv.weight.size())
print(conv2.weight.size())

print('data类型为tensor')
print(conv2.weight.data.size())
print(type(conv2.weight))
print(type(conv2.weight.data))  #variable.data

w2 = nn.Parameter(torch.randn(2 , 4) * 0.01)#这个虽然也用到nn，但不算入modu.parameters
# conv.weight.data=torch.from_numpy(np.ones(24,dtype=np.float32).reshape(6,2,2,1,1))

#module就是一个容器,module.parameters  就能调用里面的参数
#(1)实验一：实例.parameters 包含哪些变量 

class CNN2(nn.Module):#nn.module是一个基类
	def __init__(self):
		super(CNN2, self).__init__()
		self.conv1 = nn.Sequential( #创建了一个对象变量
				nn.Conv2d(
						in_channels=1,
						out_channels=16,
						kernel_size=5,
						stride=1,
						padding=2
						))
		self.w1 = nn.Parameter(torch.randn(2, 4) * 0.01)
		self.fc1 = nn.Linear(32 * 7 * 7*2, 10)
		self.cd=Variable(torch.Tensor([2]),requires_grad=True)#这个没含在modu.parameters，
		#所以想要把权重放入优化器，就要用nn.定义权重参数
	def squarr(self,x):
		return x**2
		
	def forward(self, x): #对象的参数
		# x== self.conv1(x)
		x = self.squarr(x)#
		return x #squarr，输入进来的是int，返回就是int 
print(CNN2.parameters)
modu=CNN2()

# a=modu(2)

c=Variable(torch.Tensor([2]),requires_grad=True)#如果没有[]则是定义两个数而不是2
a=modu(c)
print(a)
a.backward()
print('求导结果:',c.grad)
print(modu._parameters)#只包含了w1  nn.parameter建立的
# a.backward()
print(modu._modules)#nn.conv   nn.liner 包含在这里
print('输出整个模型',modu)
print('输出一个属性:',modu.cd)
    # def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = x.view(x.size(0), -1)
        # output = self.out(x)#logit输出，如果loss函数为mse，则还要进行softmax
        # return output#, x


# print(net.training, net.submodel1.training)
# net.train() # 将本层及子层的training设定为True
# net.eval() # 将本层及子层的training设定为False
# net.training = True # 注意，对module的设置仅仅影响本层，子module不受影响
# net.training, net.submodel1.training

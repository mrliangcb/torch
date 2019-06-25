# 第2章：torch.nn的使用
```py
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
```
# 1.常用模块  
(1)nn.parameters  
```py
a=Variable(torch.Tensor([3]), requires_grad=True) 
b=torch.nn.Parameter(torch.Tensor(a.size()), requires_grad=True)  #以variable为基类的,是variable的子类，区别是这个可以绑定到net.parameters中   

```

(2)nn.Conv2d  
```py
conv_object=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(5,5),stride=1,padding=2) #创建一个对象，含有可导variable
out_put=conv_object(variable或tensor)    #输入矩阵 输入到卷积层
conv_object.weight 
conv_object.bias 
		
```

(3)nn.linear  
```py
fc=nn.Linear(32 * 7 * 7*2, 10)   #创建全连接层
output=fc(input_var或tensor)
```
(4)nn.Sequential  
```py
#可以把多个层串联起来
self.conv1 = nn.Sequential(nn.Conv2d(),nn.ReLU(),nn.MaxPool2d(kernel_size=2))  #实例化
output=self.conv1(input) #把数据输入到集成层conv1  
```

(4)nn.Module   是一个基类  
常用于建立新的模型类  
```py
def net(nn.Module):
	def __init__(self,a):
		super(CNN2, self).__init__()
		……
		
	def forward(self, x): 
		x== self.conv1(x)
		return x
module=net(输入a参数)  #实例化模型ob
out=module(input)  #input直接传入forward中
```

(5)代价函数
criterion = nn.NLLLoss()
loss = criterion(outputs[0], Variable(torch.tensor(label).long()).cuda(0))

nn.CrossEntropyLoss() #交叉熵    nn.logSoftmax()和nn.NLLLoss()的整合
#整合如下
m = nn.LogSoftmax()
loss = nn.NLLLoss()
input=m(input)
output = loss(input, target)

nn.BCELoss()
torch.nn.MSELoss
torch.nn.L1Loss
torch.nn.PoissonNLLLoss


属性:  
```py
ob.parameters    
ob._parameters  #类里面建立的一些 nn.Parameter 会显示到这里,(类外的nn.Para不会绑定到ob.para),但类内建立的var不在这里，不会绑定到类  
ob._modules     #可以看到nn.conv  nn.linear等层

```
(6)F.softmax(x, dim=1) #代价函数用交叉熵的时候，不能用这个作为模型输出

(7)BN
m = nn.BatchNorm2d(2,affine=True)
nn.BatchNorm2d(32)

(8)激活函数用法
nn.ReLU(X)



















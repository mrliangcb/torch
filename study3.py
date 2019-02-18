#variable
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import numpy as np

x_tensor = torch.ones(3, 2)#.int()如果这里取int，则不能求requires_grad=True
y_tensor = torch.ones(3, 2)#.float  .long


print('求和:',x_tensor+y_tensor)
print('原来的x是: \n',x_tensor)

#tensor转variable
x = Variable(x_tensor, requires_grad=True) # 默认 Variable 是不需要求梯度的，所以我们用这个方式申明需要对其进行求梯度
print('data之后的值:',type(x))
print(x)
x2=x.squeeze()
print("炸成功了")
print(x)
x3=x.data.numpy()
print('x2为:',x2)
print('x3为:',x3)#(3,2)
print(x2-x3)
#squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉

y = Variable(y_tensor, requires_grad=True)
zi=y
print('y的类型:',type(y))
print(zi.view(-1))


#Variable(torch.Tensor([2]), requires_grad=True)
print('x是:\n',x,'\n y是: \n',y)
print('相加是这个: \n',x+y)
z = torch.sum(x + y) #整个tensor矩阵求和,这时候Z也是variable型，如果是多维的矩阵，可以新建一个varialb，然后逐个元素赋值
print('整个tensor矩阵求和: \n',z)
# z=torch.sum(x+y,0)#按列求和
# print('按列求和: \n',z)

print('tensor值',z.data)
print('通过sum得到的：',z.grad_fn)

#通过.grad我们得到了 x 和 y 的梯度，这里我们使用了 PyTorch 提供的自动求导机制
z.backward()
print('沿着x的梯度:',x.grad)
print('沿着y的梯度:',y.grad)


x = np.arange(-3, 3.01, 0.1)#画出3到3.01回见的X点
print(x)
y = x ** 2#整个矩阵 二次方
plt.plot(x, y)#放入两个矩阵
plt.plot(2, 4, 'ro')
plt.show()


# tensor.t()是转置
# tensor.mul(tensor)是对应元素相乘
# tensor.mm(tensor) 是横乘以竖，矩阵相乘
# variable的乘法跟上面一样

# x = Variable(tensor, requires_grad=False) //默认是false，输入都是false,权重矩阵才是True，表示输出y依赖于权重矩阵W

# variable.data变成tensor
# tensor.data还是tensor
# tensor才有.numpy()
# S=np.zeros(shape=(row,row,col),dtype=np.float)
	# S[0,0,0]=1
# variable.view 变换维度

# 变量.requires_grad   会返回是否可导
# 变量.is_leaf 返回是否叶子

























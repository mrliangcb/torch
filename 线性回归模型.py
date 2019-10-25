#线性模型 一次项的
import torch
import numpy as np
from torch.autograd import Variable





torch.manual_seed(2017)
# 读入数据 x 和 y
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
					
print('x形状:',x_train.shape)#列向量(15,1)
# 画出图像
import matplotlib.pyplot as plt
#%matplotlib inline
plt.plot(x_train, y_train, 'bo')
#plt.pause(1)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
#转tensor
#w,b变量，因为都是true,false就是常量
w = Variable(torch.randn(1), requires_grad=True) # 随机初始化w，一个数
b = Variable(torch.zeros(1), requires_grad=True) # 使用 0 进行初始化b
print('w:',w.shape)#1
# 构建线性回归模型
x_train = Variable(x_train)#把样本转化为变量，常数,grad=false,variable才能和bariable计算
y_train = Variable(y_train)

def linear_model(x): #
    return x * w + b

y_ = linear_model(x_train)

plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()

def get_loss(y_, y):#计算误差
    return torch.mean((y_ - y_train) ** 2)#求平方差 平均值
	
loss = get_loss(y_, y_train)
#用到自动求导
loss.backward()
#torch.ones_like(loss)加了一样

print('w梯度:',w.grad)
print('b梯度',b.grad)
#第一次更新数据
epsilon = 1e-5
def check_gradient(f, w,x0,y, epsilon):  #
    return (f(w+epsilon,x0,y) - f(w-epsilon,x0,y))/2/epsilon
	
def all(w,x,y):
	y_= x * w + b
	return get_loss(y_,y)
	
print('梯度检查:',check_gradient(all,w,x_train ,y_train, epsilon))

w.data = w.data - 1e-2 * w.grad.data#学习率乘以梯度,如果梯度正的，就是梯度上升，那自变量就是减小
b.data = b.data - 1e-2 * b.grad.data




y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()
#接下来做十次更新
for e in range(10): # 进行 10 次更新
    y_ = linear_model(x_train)#每次都把全部值算一遍
    loss = get_loss(y_, y_train)
    
    w.grad.zero_() # 记得归零梯度
    b.grad.zero_() # 记得归零梯度
    loss.backward()
    
    w.data = w.data - 1e-2 * w.grad.data # 更新 w
    b.data = b.data - 1e-2 * b.grad.data # 更新 b 
    print('epoch: {}, loss: {}'.format(e, loss.data[0]))
y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()




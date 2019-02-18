#神经网络
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt

def plot_decision_boundary(model, x, y):#一个model就是一个lambda
	# Set min and max values and give it some padding
	x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
	y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
	h = 0.01
	# Generate a grid of points with distance h between them
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))#产生网格
	print('xx是什么:',xx)
	# Predict the function value for the whole grid
	Z = model(np.c_[xx.ravel(), yy.ravel()])#使用模型lambda
	print('看一下Z:\n',Z)
	Z = Z.reshape(xx.shape)
	#print(Z.shape)#输出的是1,0
	# Plot the contour and training examples
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)#画出等高值
	plt.ylabel('x2')
	plt.xlabel('x1')
	plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)


np.random.seed(1)  #种子就是产生随机数
m = 400 # 样本数量
N = int(m/2) # 每一类的点的个数
D = 2 # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8') # label 向量，0 表示红色，1 表示蓝色
a = 4

for j in range(2):
	ix = range(N*j,N*(j+1))
	t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
	r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
	x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
	y[ix] = j

plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)

#逻辑回归做一下
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))

optimizer = torch.optim.SGD([w, b], 1e-1)

def logistic_regression(x):#单层神经网
    return torch.mm(x, w) + b

criterion = nn.BCEWithLogitsLoss()
for e in range(100):
    out = logistic_regression(Variable(x))
    loss = criterion(out, Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 20 == 0:
        print('epoch: {}, loss: {}'.format(e+1, loss.data[0]))

def plot_logistic(x):
	x = 0Variable(torch.from_numpy(x).float())
	out = torch.sigmoid(logistic_regression(x))
	out = (out > .5) * 1
	print('模型输出',out) #rensor，1,0
	print('输出:',out.data.numpy())
	return out.data.numpy()
	


plot_decision_boundary(lambda x: plot_logistic(x), x.numpy(), y.numpy())#(1/0,x坐标,y坐标)
plt.title('logistic regression')
plt.show()
	

#神经网路mlp
print('用神经网络\n')
# 定义两层神经网络的参数#多层逻辑回归
w1 = nn.Parameter(torch.randn(2, 4) * 0.01) # 输入为2个特征,隐藏层神经元个数 4
b1 = nn.Parameter(torch.zeros(4))

w2 = nn.Parameter(torch.randn(4, 1) * 0.01)
b2 = nn.Parameter(torch.zeros(1))

print('W1是什么',w1.shape)#[2,4]
# 定义模型
def two_network(x):
    x1 = torch.mm(x, w1) + b1
    x1 = F.tanh(x1) # 使用 PyTorch 自带的 tanh 激活函数
    x2 = torch.mm(x1, w2) + b2
    return x2

optimizer = torch.optim.SGD([w1, w2, b1, b2], 1.)

criterion = nn.BCEWithLogitsLoss()
	
# 我们训练 10000 次

print('x是什么:',x.shape)#[400,2]

for e in range(10000):
    out = two_network(Variable(x))
    loss = criterion(out, Variable(y))
    optimizer.zero_grad()
    loss.backward()
    print("w1是多少:",w1)
    optimizer.step()
    print("优化后的w1是多少:",w1)
    if (e + 1) % 1000 == 0:
        print('epoch: {}, loss: {}'.format(e+1, loss.data[0]))
def plot_network(x):
    x = Variable(torch.from_numpy(x).float())
    x1 = torch.mm(x, w1) + b1
    x1 = torch.tanh(x1)
    x2 = torch.mm(x1, w2) + b2
    out = torch.sigmoid(x2)
    out = (out > 0.5) * 1
    return out.data.numpy()
print('y.numpy()是这个:',y.numpy().shape)#400*1的矩阵



plot_decision_boundary(lambda x: plot_network(x), x.numpy(), y.numpy())
plt.title('2 layer network')
plt.show()

#在平面上的情况
#逻辑回归只能是一条线，一次，二次，三次等函数
#神经网络似乎有多条线可以用





	
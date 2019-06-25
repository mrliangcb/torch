#线性模型，多项式的
# 定义一个多变量函数
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

w_target = np.array([0.5, 3, 2.4]) # 定义参数，定义三个，一维
k_target = np.array([[1, 2, 3],
					[4,5,6]])

b_target = np.array([0.9]) # 定义参数，定义一个偏置项

f_des = 'y = {:.2f} + {:.2f} * x + {:.2f} * x^2 + {:.2f} * x^3'.format(
    b_target[0], w_target[0], w_target[1], w_target[2]) # 打印出函数的式子

print(f_des)

# 画出样本和label
x_sample = np.arange(-3, 3.1, 0.1)#样本的np，间隔为0.1
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3
print('y是:',y_sample.shape)#61个元素
# print('测试:',k_target*w_target) 两个矩阵相乘
#[]矩阵乘以一个数

plt.plot(x_sample, y_sample, 'ro-',label='real curve',markerfacecolor='red')

plt.legend()
plt.close()
# plt.show()



# 构建数据 x 和 y
# x 是一个如下矩阵 [x, x^2, x^3]
# y 是函数的结果 [y]

x_train = np.stack([x_sample ** i for i in range(1, 4)], axis=1)#[第一个元素是[所有一次方],第二个元素是[所有二次方]，第三个元素]，按照列方向拼接
x_train = torch.from_numpy(x_train).float() # 转换成 float tensor
#print('x形状:',x_train.shape)

y_train = torch.from_numpy(y_sample).float().unsqueeze(1) # 转化成 float tensor

# 将 x 和 y 转换成 常数型Variable
x_train = Variable(x_train)
y_train = Variable(y_train)

# 定义参数和模型，变量型
w = Variable(torch.randn(3, 1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)

def multi_linear(x):
    return torch.mm(x, w) + b

# 画出更新之前的模型
y_pred = multi_linear(x_train)
#(61*3)  *  (3*1)

print('x形状:',x_train.shape)#61*3
plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
plt.legend()
plt.title(u"初始化的情况")
# plt.show()
plt.close()

def get_loss(y_, y):#计算误差
    return torch.mean((y_ - y_train) ** 2)#求平方差 全部训练样本的平均值

loss = get_loss(y_pred, y_train)
# w.data=w.data*100 #让variable的初始值改变  这里的改变会影响到后面求导的
print('损失值：',loss)
# w[0,0]=1  #inplace操作 直接给variable赋值都是不好的，w=w*0.2 也会有问题，
#会导致后面w.data错误，求导值也是none，因为w由variable变成了tensor


# w.data[0,0]=w.data[0,0]*0.2 #这样就可以的
print('类型',type(w),type(w.data))#左边应给是variable,右边是tensor
print('权重',w)
print('是否可导',w.requires_grad,w.data.requires_grad) #

# 自动求导
loss.backward()
# 查看一下 w 和 b 的梯度
print('w的梯度:',w.grad)  #因为求导之后2x^2这种，基于x=2这一点的梯度值
print('w的梯度',b.grad)


# 更新一下参数
w.data = w.data - 0.001 * w.grad.data
b.data = b.data - 0.001 * b.grad.data

y_pred = multi_linear(x_train)

plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
plt.title(u'first update')
plt.legend()
plt.show()

for e in range(100):#更新一百次
    y_pred = multi_linear(x_train)
    loss = get_loss(y_pred, y_train)
    
    w.grad.data.zero_()
    b.grad.data.zero_()
    loss.backward()
    
    # 更新参数
    w.data = w.data - 0.001 * w.grad.data
    b.data = b.data - 0.001 * b.grad.data
    if (e + 1) % 20 == 0:
        print('epoch {}, Loss: {:.5f}'.format(e+1, loss.data[0]))

# 画出更新之后的结果
y_pred = multi_linear(x_train)

plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
plt.title(r'final result')
plt.legend()
plt.show()











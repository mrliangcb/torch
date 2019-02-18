#求导问题
import torch
import numpy as np
from torch.autograd import Variable

b=1
for i in range(1,0,-1):
	print("i是:",i)
	
	
	
	
#注意给
x2 = Variable(torch.FloatTensor(np.array([[3,6,9],[1,2,3]])), requires_grad=True)
x3 = Variable(torch.FloatTensor(np.array([[1,1,1],[1,1,1]])), requires_grad=True)
x2[:,1]=x3[:,1]  #variable矩阵的一个元素取出来，也是variable，可以给variable的单个元素赋值（可以赋单个item()或者单个variable）
print("给variable选择区域赋值",x2)#也可以给variable矩阵的一个区域赋值另一个variable矩阵
#

x2 = Variable(torch.FloatTensor(np.array([[3,6,9],[1,2,3]])), requires_grad=True)
y2=x2/3
z5=torch.mean(y2,dim=0) #0就是按列求平均，1就是按行
print("求平均",z5)


#实验求迹 对角线之和   如果是二维的话(7,8,9去掉)就是1+5=6
x3 = Variable(torch.FloatTensor(np.array([[1,2,3],[4,5,6],[7,8,9]])), requires_grad=True)
y3=np.trace(x3.data.numpy())
print(y3)
print("迹为",type(y3))
#不能variable除以numpy，只能除以python float型(也就是variable.item())  
z4=torch.mean(x3/y3.item())#可以赋回给自己，也可以赋给别人 之前用x3=torch.mean()是不行的，出现叶子已经在图中的报错，避免赋值给自己
print("variable矩阵除以一个item",x3)
z4.backward()
print("成功")
x = Variable(torch.Tensor([2]), requires_grad=True)#x自变量，给一个2，不是1维2列的意思，如果只当做tensor而不是变量，则常数不可导
#x = Variable(torch.Tensor(np.array([2])), requires_grad=True)#表示建立了一个值为2的变量
# x = Variable(torch.Tensor(np.zeros(shape=(1))), requires_grad=True)
#上面三种定义都可以求导，np怎么定义都行，tensor就要[2]和2区分
y = x + 2
z = y ** 2 + 3
# zz = Variable(torch.Tensor([2]), requires_grad=True)
# zz=z #就算设置一个中间变量来装，也可以求导的

print(z)
#z=(x+2)平方,+3
#求导就等于2x
print("开始求导")
z.backward() #第四行的[2]对这里有影响，2和[2]都是创建一个初始化为2的tensor，但前者是常量，后者是变量，只是赋初始值2
print(x.grad)
print("Z对自己的求导",z.grad)
print("结束求导")




#上面是标量式的
#创建一个变量，并且赋值
x = Variable(torch.randn(10, 20), requires_grad=True)
y = Variable(torch.randn(10, 5), requires_grad=True)
w = Variable(torch.randn(20, 5), requires_grad=True)
out = torch.mean(y - torch.matmul(x, w)) # torch.matmul 是做矩阵乘法


print("求平均",out)
#y-x*w  三个都是变量
out.backward()
print(x.grad.shape) #求导之后为10行20列，每个变量都要求导，同理，y.grad  w.grad都可以求



m = Variable(torch.FloatTensor([[2, 3]]), requires_grad=True) # 构建一个 1 x 2 的矩阵
n = Variable(torch.zeros(1,2)) # 构建一个相同大小的 0 矩阵
print(m)
print(n)
n[0, 0] = m[0, 0] ** 2  #一行两列的，所以第一个元素是00，第二个是0,1。如果是zeros(2)的话，就直接n[0],n[1]
n[0, 1] = m[0, 1] ** 3
x2 = Variable((torch.ones(1,2)), requires_grad=False)
n=n+x2#相当于加了常数，求导不影响
#n=torch.sum(n+x2) #不行
print(n)
n.backward(torch.ones_like(n))
print('m梯度',m.grad)


print('多次求导')
x = Variable(torch.FloatTensor([3]), requires_grad=True)

y = x * 2 + x ** 2 + 3
print(y)
y.backward(retain_graph=True)
print(x.grad)
y.backward()
print(x.grad)

print('小练习:雅克比')
x = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
k = Variable(torch.zeros(2))

k[0] = x[0] ** 2 + 3 * x[1]
k[1] = x[1] ** 2 + 2 * x[0]

print(k)
j = torch.zeros(2, 2)

k.backward(torch.FloatTensor([1, 0]), retain_graph=True)
j[0] = x.grad.data

x.grad.data.zero_() # 归零之前求得的梯度

k.backward(torch.FloatTensor([0, 1]))
j[1] = x.grad.data
print(j)

# 链式求导
# 如果对非标量求导的话，需要指定grad_tensors  这个和输出y的形状一定要相同















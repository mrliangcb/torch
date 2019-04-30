#参考 https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308
import torch
import numpy as np
#leaf variable和普通variable
x = torch.autograd.Variable(torch.Tensor([[1, 2, 3]]),requires_grad=True)  #是叶子变量 leaf variable
x1 = torch.autograd.Variable(torch.Tensor([[2,1], [5,1], [7,1]]),requires_grad=True)
print()

y = x + 1  # 非叶子变量  variable
print('x是否可导:',x.requires_grad)

#还有in-place变量 
# x += 1  # in-place  仍是自己的，就是in place  不能用于可导的x
y = x + 1 # not in place 新建了一个变量

y=torch.mean(y)
# y.backward()
print(y)

#注意
#1.pytorch 不允许对leaf variable进行in place操作
#2.也就是说x=x+1是错的,y=y+1是对的，因为x是模型，这样做我就改变了模型了
#3.x2 = x.clone()
#		x2 += 1 把模型variable克隆出来，然后再inplace操作


# 变量.requires_grad   会返回是否可导
# 变量.is_leaf 返回是否叶子
#产生非叶子节点的方法(1)无中生有z=x+y, x,y是自定义的，z是未见过的  (2)自定义一个z，然后z2=z.clone()
#特别是中间变量的问题
#X,Y是建立的也自变量，如果要做一个矩阵shape=(1,3),分别放均值，和，方差
	#方法，新建一个
	# S0=np.zeros(shape=(1,3),dtype=np.float)
	# variable(torch.FloatTensor(S0),requires_grad=True)
	#
def con(a,b):
	c=torch.autograd.Variable(torch.Tensor([1, 2]))#创建一个矩阵，第一位与第二位分别装两个可导变量，中介矩阵不能是可导的，因为可导的不能修改，
	#可导的都是基本元，只能通过初始化来赋值
	print(torch.mm(a,b))#得到二维的
	c[0]=torch.mm(a,b)[0,0]
	c[1]=torch.mm(a,b)[0,1]
	return c
	
d=con(x,x1)
dd=torch.sum(d)
dd.backward()
print('结果阵',dd)
print(x.grad)
print(x1.grad)



# 定义了[x1,x2,x3] 
# 还有[[x4,x5],
	# [x6,x7],
	# [x8,x9]]
# out=(x1*x4+x2*x6+x3*x8)+(x1*x5+……)
	# =x1(x4+x5)+……
# 所以d(out)/d(x1)=x4+x5
	
	
	
	
	
	
	




















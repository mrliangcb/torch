import torch
import numpy as np
from torch.autograd import Variable
import numpy as np
from scipy.fftpack import fft,ifft,irfft,rfft
import model as m
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

def cxcorr(v,a):
	a=a.data.numpy();
	v=v.data.numpy();
	nom = np.linalg.norm(a[:])*np.linalg.norm(v[:])
	return irfft(rfft(a)*rfft(v[::-1]))/nom
	
def cross_correlation(Y):
	
	row=Y.shape[0] #空间维度
	col=Y.shape[1] #时间维度
	
	# print(row,col)
	S0=np.zeros(shape=(2*col-1,row,row),dtype=np.float64)
	
	S=Variable(torch.FloatTensor(S0),requires_grad=True)
	S2=S.clone()#
	for tn in range(2*col-1):#设置延时 位移量为多少
		tnn=tn-col+1#位移量与下标关系,下标为0到2T-1，位移是-T到T
		
		for i in range(row):
			for j in range(row):
				# if tn==0:#tn为0的时候就直接乘以
					# for t in range(col):
						# S2[i,j,tn]+=Y[i,t]*Y[j,t]
						# print("贡献了多少:",S2[i,j,tn],i,j,tn)
					# 如果这里直接赋值给S的话，S是leaf variable，不能改变的，
					# clone出来的是非叶子节点，可以改变，所以拿来做中间变量

				# else:#tn不为0的时候要移位
					# print("flag清零了没有:",flag2)
					for t in range(col):#移位之后相乘的和
						#print("t:",t,"tnn:",tnn,"col:",col,(t-tnn)<col and (t-tnn)>=0)
						if (t-tnn)<col and (t-tnn)>=0:
							S2[tn,i,j]+=Y[i,t]*Y[j,t-tnn] #用整数来装variable
							#print("贡献了多少:",Y[i,t].data,Y[j,t-tnn].data,S2[i,j,tn].data,i,j,t-tnn)
						#得到了tn下的一个i,j和
						#得到了tn下的全部i,j和，也就是一个二维矩阵
						 #可以用整数赋值给variable
					
	
	
	for tn in range(2*col-1):#col就是tn
		# #先求迹，一个数
		S_tr=np.trace(S2[tn,:,:].data.numpy()) #inplace 操作
		#print("迹为多少:",S_tr)#27(3*9)和54(3*18)
		S2[tn,:,:]=torch.div(S2[tn,:,:],S_tr.item())#这整个过程中S2因为是克隆出来的，所以一直都是非叶子变量
	return S2

def mean_tn(S):#沿着时间轴2tn-1求平均,
	print("S输入维度:",S.shape)
	result=torch.mean(S,dim=0)#注意，第一维是最外面的一维，也就是时间维
	print("result结果维度",result.shape)
	return result
	
def var_tn(S):
	print("var处理前的大小",S.shape)
	result=torch.var(S,dim=0)
	print("var处理后的大小",result.shape)
	return result 

data_ar=np.ones(shape=(30,10),dtype='float32')
x_tensor = torch.FloatTensor(data_ar)

W_ar=np.ones(shape=(30,30),dtype='float64')
W_tensor = torch.FloatTensor(W_ar)#[[1,2,3],[4,5,6],[7,8,9]]

x = Variable(x_tensor, requires_grad=False) 
W = nn.Parameter(W_tensor, requires_grad=True) 
# print("判断是否可导",W.requires_grad)
#y是非叶子变量
y=W.mm(x)#tensor也能和variable计算  合成一个矩阵的话可导
print("输入矩阵:",y)
#套上torch.mean之后是一个数，而且是meanbackward

#y.backward() #y=torch.mean(W*W) 这样就可导，因为输出是单个变量,而且y是非叶子变量，像x=(含x)就不行，这种是inplace操作，对也自变量不可以inplace操作
#如果y=W*W则不可直接导，多个变量，等之后计算loss函数的时候，就变成一个变量了


# print("是否叶子",y.is_leaf)
S=cross_correlation(y) 
Sm=mean_tn(S)
print("Sm为",Sm)
Sv=var_tn(S)
print("Sv为",Sv)
Gm=0.5*(Sm+1)
Gv=0.5*(Sv+1)
Gm=Gm.view([1,1,30,30])#增加时间通道维度
Gv=Gv.view([1,1,30,30])
input=torch.cat((Gm,Gv),1)#沿着时间通道联立
input=torch.cat((input,input),0)
print("input的大小",input)


# total=torch.mean(Sm)#不加dim则对整个tensor求平均，加dim就沿着特定方向求
# total.backward()

# (N,Cin,H,W) ，N NN表示batch size，Cin C_{in}C in   
 # 表示channel个数，H HH，W WW分别表示特征图的高和宽。   padding是在卷积前补零的
 #padding=1表示四周都补1
 #输出大小=(输入边长+2*padding-核边长)/步长    +1
class CNN2(nn.Module):
	def __init__(self):
		super(CNN2, self).__init__()
		self.conv1 = nn.Sequential(
				nn.Conv2d(
						in_channels=2,
						out_channels=8,
						kernel_size=4,
						stride=1,
						padding=1 #默认是valid
						),
				nn.LeakyReLU(negative_slope=0.2),
				nn.MaxPool2d(kernel_size=8)
				)
				
		self.conv2 = nn.Sequential(
				nn.Conv2d(8, 16, 2, 1, 1), 
				nn.LeakyReLU(0.2), 
				nn.MaxPool2d(4))
		self.out = nn.Linear(16, 2)

	def forward(self, x):
		print("输入的大小:",x.shape)#[1,2,3,3]
		x = self.conv1(x)#x是一个variable
		print("输出大小",x.shape)##[1,8,3,3]
		x = self.conv2(x)
		print("最终输出:",x.shape,)#1,16,1,1
		x = x.view(x.size(0), -1)
		output = self.out(x)#logit输出，如果loss函数为mse，则还要进行softmax
		return output#, x
net=CNN2()

net.train()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
optimizer.zero_grad()

criterion = nn.CrossEntropyLoss()
outputs = net(input)
print("网络输出",outputs)
loss = criterion(outputs, Variable(torch.tensor([0,1])))
print("loss值:",loss)
loss.backward()
print("W的求导",W.grad)

optimizer.step()
print("现在的W值：",W)
outputs = net(input)
criterion(outputs, Variable(torch.tensor([0,1])))
print("loss第二次值:",loss)




# test=np.arange(24).reshape((4,2,3))#   (2,3,4)表示3行4列，时间轴为2   求平均的时候，dim=0对应时间轴
# test = torch.FloatTensor(test)
# test = Variable(test, requires_grad=False) 

# test2=np.arange(24).reshape((4,2,3))#   (2,3,4)表示3行4列，时间轴为2   求平均的时候，dim=0对应时间轴
# test2 = torch.FloatTensor(test)
# test2 = Variable(test, requires_grad=False) 
# all=torch.cat((test,test2),0) #连接两个variable矩阵

# print("连接后的大小",all.shape)
# print(test)
# print(mean_tn(test))

# print(W.grad)

# print("判断是否可导",S.requires_grad)#因为S依赖了一些可导变量
# S_final=torch.sum(S)
# print("判断是否可导",S_final.requires_grad)

#torch.autograd.grad(outputs=y,inputs=W,grad_outputs=torch.ones_like(W))
# S_final.backward()
# S[0,0,0].backward(torch.ones_like(W[0,0]))


# https://fanyublog.com/2015/11/16/corr_python/
# 讲了线性相关和互相关


	
	

# https://blog.csdn.net/ANNILingMo/article/details/78006227
# https://blog.csdn.net/TH_NUM/article/details/83088915


# https://blog.csdn.net/a19990412/article/details/83904476#padding_21
# 将torchpadding





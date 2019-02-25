import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import read_data

print("第二次修改")
pra=1#=0表示用小矩阵在测试  =1表示对116*172
home=0


if home==0:
	txt_path=r'D:\脑科学\NYU\ADHD200_AAL_TCs_filtfix\NYU\0010002\snwmrda0010002_session_1_rest_1_aal_TCs.1D'
else:
	txt_path=r'snwmrda0010002_session_1_rest_1_aal_TCs.1D'



# print("判断是否可导",W.requires_grad)
#y是非叶子变量
#tensor也能和variable计算  合成一个矩阵的话可导

#套上torch.mean之后是一个数，而且是meanbackward

#y.backward() #y=torch.mean(W*W) 这样就可导，因为输出是单个变量,而且y是非叶子变量，像x=(含x)就不行，这种是inplace操作，对也自变量不可以inplace操作
#如果y=W*W则不可直接导，多个变量，等之后计算loss函数的时候，就变成一个变量了


# print("是否叶子",y.is_leaf)




# total=torch.mean(Sm)#不加dim则对整个tensor求平均，加dim就沿着特定方向求
# total.backward()

# (N,Cin,H,W) ，N NN表示batch size，Cin C_{in}C in   
 # 表示channel个数，H HH，W WW分别表示特征图的高和宽。   padding是在卷积前补零的
 #padding=1表示四周都补1
 #输出大小=(输入边长+2*padding-核边长)/步长    +1
class CNN2(nn.Module):
	def __init__(self):
		super(CNN2, self).__init__()
		if pra==0:
			self.W = nn.Parameter(torch.ones(30,30)*0.1, requires_grad=True) #有self和没self
		if pra==1:
			self.W = nn.Parameter(torch.ones(116,116)*0.1, requires_grad=True) #
		self.conv1 = nn.Sequential(
				nn.Conv2d(
						in_channels=2,
						out_channels=8,
						kernel_size=4,
						stride=1,
						padding=1 #默认是valid
						),
				nn.LeakyReLU(negative_slope=0.2),
				nn.MaxPool2d(kernel_size=8)#115/8=14  14*14
				)
				
		self.conv2 = nn.Sequential(
				nn.Conv2d(8, 16, 2, 1, 1), 
				nn.LeakyReLU(0.2), 
				nn.MaxPool2d(4))#[1,16,3,3]
		if pra==0:
			self.out = nn.Linear(16, 2)
		if pra==1:
			self.out = nn.Linear(16*3*3, 2)
	def read_1d(self,path):
		with open(path, mode='rt') as f:
					lines = f.readlines()
					# print(lines[1])
					# print(type(lines))#lines是一个list
					# print(len(lines))#长度为173
					# print(len(lines[0]))#一行是一个str
					# print(lines[0])
					tokens = [i.rstrip('\n').split('	')[2:] for i in lines]
					# print('每行长度',len(tokens[1]))#116
					# print('第一行内容',tokens[0])
					# print('第二行内容',tokens[1])
					#变成矩阵
					matrix=np.array(tokens[1:])
					matrix=matrix.transpose(1,0)
					# print(type(matrix[0][0]))#str类型
					matrix1=matrix.astype(np.float32)
					# print(type(matrix1[0][0]))#float32类型
					# print(matrix1[0][0].dtype)#float32类型
					return matrix1
		
	def cross_correlation(self,Y): 
		#要写上self  要把这个输入当成是tensor，tensor盒子，整个过程运算都是用tensor，不能用variable，更加不能新建
		#当我们输入variable的时候，输出就自然是variable了
		print("输入大小:",Y.shape)
		row=Y.shape[0] #空间维度
		col=Y.shape[1] #时间维度
		
		# print(row,col)
		
		S=torch.Tensor(2*col-1,row,row).zero_()
		
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
						cal=torch.Tensor(col).zero_()
						for t in range(col):#移位之后相乘的和
							
							#print("t:",t,"tnn:",tnn,"col:",col,(t-tnn)<col and (t-tnn)>=0)
							if (t-tnn)<col and (t-tnn)>=0:
								cal[t]=Y[i,t]*Y[j,t-tnn] #用整数来装variable，如果赋值S2.data给自己，则只是给一数没有给变量节点
						S[tn,i,j]=cal.sum()
						
								#print("贡献了多少:",Y[i,t].data,Y[j,t-tnn].data,S2[i,j,tn].data,i,j,t-tnn)
							#得到了tn下的一个i,j和
							#得到了tn下的全部i,j和，也就是一个二维矩阵
							 #可以用整数赋值给variable
		S_mean=torch.Tensor(2*col-1,row,row).zero_()
		for tn in range(2*col-1):#col就是tn
			# #先求迹，一个数
			S_tr=torch.trace(S[tn,:,:]) #
			#print("迹为多少:",S_tr)#27(3*9)和54(3*18)
			S_mean[tn,:,:]=torch.div(S[tn,:,:],S_tr)#这整个过程中S2因为是克隆出来的，所以一直都是非叶子变量
		return S_mean
	
	def mean_tn(self,S):#沿着时间轴2tn-1求平均,
		result=torch.mean(S,dim=0)#注意，第一维是最外面的一维，也就是时间维
		return result

	def var_tn(self,S):
		result=torch.var(S,dim=0)
		return result 
	
	def forward(self, x):#输入一串list[path1,path2,……]
		
		for i in range(len(x)):
			print("输入类型",type(x[i]))
			matrix_i=self.read_1d(x[i])
			matrix_ii=Variable(torch.Tensor(matrix_i))
			y=torch.mm(self.W,matrix_ii)
		# print("试一下y求导")
		# y1=y.mean()
		# y1.backward(retain_graph=True)
		# print("y求导成功")
		# print("y的大小:",y.shape)  #[30,20]
			S=self.cross_correlation(y) 
		# print("S的值",S)
		# print("对S求导")
		# S1=S.mean()
		# S1.backward(retain_graph=True)
		# print("S求导成功")
			Sm=self.mean_tn(S)
			Sv=self.var_tn(S)
			Gm=0.5*(Sm+1)
			Gv=0.5*(Sv+1)
			if pra==0:
				Gm=Gm.view([1,1,30,30])#增加时间通道维度
				Gv=Gv.view([1,1,30,30])
			if pra==1:
				Gm=Gm.view([1,1,116,116])#增加时间通道维度
				Gv=Gv.view([1,1,116,116])
			input=torch.cat((Gm,Gv),1)#沿着时间通道联立
			if i==0:
				input_tensor=input
			else :
				input_tensor=torch.cat((input_tensor,input),0)
		print("到这里")
		print("输入的大小:",input_tensor.shape)
		#print("输入的大小:",input.shape)#[1,1,3,3]
		out = self.conv1(input_tensor)#x是一个variable
		print("输出大小",out.shape)##[1,8,3,3]
		out1 = self.conv2(out)
		print("最终输出:",out.shape)#1,16,1,1
		out2 = out1.view(out1.size(0), -1)
		output = self.out(out2)#logit输出，如果loss函数为mse，则还要进行softmax
		return output#, x
		
if pra==0:
	data_ar=np.ones(shape=(30,20),dtype='float32')
	x_tensor = torch.Tensor(data_ar)
# if pra==1:
	# x_tensor = torch.Tensor(matrix1)#先将np转为tensor和variable，116*172
# x = Variable(x_tensor, requires_grad=False) 
		
#模型实例
net=CNN2()
net.train()
#把模型参数加到优化器
optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
criterion = nn.CrossEntropyLoss()

batch=8
epoch=3
num=0

# def make_batch(data_list,label_list,batch,num):
	# data=data_list[num:(num+batch)]
	# label=label_list[num:(num+batch)]
	# a=[read_1d(i) for i in data]
	# for j in range(0,len(a),1): 
		# if j==0:
			# data_batch =torch.Tensor(a[j]).view([1,1,116,-1])
		# if j>0:
			# data_b=torch.Tensor(a[j]).view([1,1,116,-1])
			# data_batch=torch.cat((data_batch,data_b),0)
	# data_batch=Variable(data_batch)
	# label_batch=Variable(torch.Tensor(label))
	# return data_batch,label_batch
# Gm=Gm.view([1,1,116,116])#增加时间通道维度
			# Gv=Gv.view([1,1,116,116])
		# input=torch.cat((Gm,Gv),1)#

data_list,label_list = read_data.get_list(epoch)#生成n个epoch的数据

#data_tensor,label_tensor = make_batch(data_list,label_list,batch,num)
# print(data_tensor.shape)
outputs = net(data_list)

# for i in range(100):
	# data_tensor,label_tensor = make_batch(data_list,label_list,batch,num)
	# optimizer.zero_grad()
	# outputs = net(data_tensor)
	# loss = criterion(outputs, label_tensor)
	# loss.backward()
	# optimizer.step()
	
				
					
					
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





import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import read_data

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('电脑设备:',device)

print("第五次修改")
pra=0#=0表示用小矩阵在测试  =1表示对116*172
home=0
gpu=False

# if home==0:
	# txt_path=r'D:\脑科学\NYU\ADHD200_AAL_TCs_filtfix\NYU\0010002\snwmrda0010002_session_1_rest_1_aal_TCs.1D'
# else:
	# txt_path=r'snwmrda0010002_session_1_rest_1_aal_TCs.1D'
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

	def forward(self, x):#输入[batch,2,116,116]
		print("输入的大小:",x.shape)
		out = self.conv1(x)#x是一个variable
		print("conv1:",out.shape)##[1,8,3,3]
		out1 = self.conv2(out)
		print("conv2:",out1.shape)#1,16,1,1
		out2 = out1.view(out1.size(0), -1)
		output = self.out(out2)#logit输出，如果loss函数为mse，则还要进行softmax
		return output#, x
		
# if pra==0:
	# data_ar=np.ones(shape=(30,20),dtype='float32')
	# x_tensor = torch.Tensor(data_ar)
# if pra==1:
	# x_tensor = torch.Tensor(matrix1)#先将np转为tensor和variable，116*172
# x = Variable(x_tensor, requires_grad=False) 
def read_1d(path):
		with open(path, mode='rt') as f:
			lines = f.readlines()
			tokens = [i.rstrip('\n').split('	')[2:] for i in lines]
			matrix=np.array(tokens[1:])
			matrix=matrix.transpose(1,0)
			matrix1=matrix.astype(np.float32)
			return matrix1
					
def cross_correlation(Y): #计算三维相关矩阵
		print("输入大小:",Y.shape)
		row=Y.shape[0] #空间维度
		col=Y.shape[1] #时间维度
		if gpu:S=torch.Tensor(2*col-1,row,row).zero_().cuda(0)
		else:S=torch.Tensor(2*col-1,row,row).zero_()#.cuda(0)
		for tn in range(2*col-1):#设置延时 位移量为多少
			tnn=tn-col+1#位移量与下标关系,下标为0到2T-1，位移是-T到T
			for i in range(row):
				for j in range(row):
					
					for t in range(col):#移位之后相乘的和
						if (t-tnn)<col and (t-tnn)>=0:
							S[tn,i,j]=S[tn,i,j]+Y[i,t]*Y[j,t-tnn]
		if gpu:S_mean=torch.Tensor(2*col-1,row,row).zero_().cuda(0)
		else:S_mean=torch.Tensor(2*col-1,row,row).zero_()
		for tn in range(2*col-1):#col就是tn
			S_tr=torch.trace(S[tn,:,:]) #
			S_mean[tn,:,:]=torch.div(S[tn,:,:],S_tr)
		print("计算相关矩阵完成")
		return S_mean
def mean_tn(S):#沿着时间轴2tn-1求平均,
		result=torch.mean(S,dim=0)#注意，第一维是最外面的一维，也就是时间维
		return result
def var_tn(S):
	result=torch.var(S,dim=0)
	return result 
#建立y=wx的权重
if pra==0:
	if gpu:W = nn.Parameter(torch.ones(30,30)*0.1, requires_grad=True).cuda(0)#有self和没self
	else:W = nn.Parameter(torch.ones(30,30)*0.1)#.cuda(0)#有self和没self
else:
	if pra==1:W = nn.Parameter(torch.ones(116,116)*0.1, requires_grad=True)#.cuda(0) #
	else:W = nn.Parameter(torch.ones(116,116)*0.1, requires_grad=True).cuda(0) #
	
	
def make_batch_var(path_list):
	for i in range(1):
		#读取该路径下的文件，得到numpy矩阵
		#matrix_i=read_1d(path)
		#matrix_ii=Variable(torch.tensor(matrix_i)).cuda(0)
		matrix_ii=Variable(torch.ones(30,20))#.cuda(0)
		y_i=torch.mm(W,matrix_ii)#cuda运算
		S=cross_correlation(y_i) #得到一个三维矩阵var
		Sm=mean_tn(S)#得到116*116
		Sv=var_tn(S)
		Gm=0.5*(Sm+1)
		Gv=0.5*(Sv+1)
		if pra==0:
			Gm=Gm.view([1,1,30,30])#增加时间通道维度
			Gv=Gv.view([1,1,30,30])
		if pra==1:
			Gm=Gm.view([1,1,116,116])#增加时间通道维度
			Gv=Gv.view([1,1,116,116])
		input=torch.cat((Gm,Gv),1)#沿着时间通道联立，于是时间深度为2
		if i==0:
			input_tensor=input
		else :
			input_tensor=torch.cat((input_tensor,input),0)#在batch通道连接
	return input_tensor
		
#模型实例
net=CNN2()
if gpu:net=net.cuda(0)
select_cuda=0


net.train()
#把模型参数加到优化器
print("模型参数:",net.parameters())
optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
optimizer2 = torch.optim.SGD([W], lr = 0.5)

criterion = nn.CrossEntropyLoss()

batch_size=1
epoch=3
num=0

if home==1:path=r'E:\deep_learning\adhd\training_data'
else : path=r'D:\脑科学\NYU\training_data'
data_list,label_list = read_data.get_list(path,epoch)#生成n个epoch的数据


for i in range(1):
	#先取数据
	x=data_list[i:i+batch_size]
	#取出一个batch的variable
	x_batch=make_batch_var(x)#传入一串路径，输出[batch,2,116,116]的variable
	y=label_list[i:i+batch_size]
	label=np.array(y).astype(np.int64)
	optimizer.zero_grad()
	outputs = net(x_batch)#输入路径
	print('输出是什么:',outputs)
	print("运行loss")
	if gpu:loss = criterion(outputs, Variable(torch.tensor([0])).cuda(0))
	else:loss = criterion(outputs, Variable(torch.tensor(label)))
	print("loss值是多少：",loss)
	loss.backward()
	optimizer.step()
	optimizer2.step()
	print("权重求导",W.grad)
	print("优化后的权重",W)
	if gpu:
		train_pre= torch.max(outputs,1)[1].cpu().data.numpy()
	else:
		train_pre= torch.max(outputs,1)[1].data.numpy()
	train_acc=np.sum(train_pre==label)/batch_size
	print("准确率:{:.3%}".format(train_acc))




#cross_validation写法
import torch.nn as nn
import math
import torch

# 高级类  nn.module   像net  conv pool这些 都是继承了module
# 低级类  nn.Parameter 没有module属性的
class mymodule(nn.Module):
	def __init__(self):
		super(mymodule,self).__init__()
		self.weight=nn.Parameter((torch.rand(90,90)), requires_grad=True)
	def forward(self,x):
		return x


class Net(nn.Module):  #一个模块(字典)，下面包含了子模块conv2d(变量名为key)
	def __init__(self):
		super(Net, self).__init__()
		self.convE1 = nn.Conv2d(1, 16, (1,300)) #自己默认 weight.data.uniform  bias.data.uniform
		nn.init.normal_(self.convE1.weight, std=math.sqrt(2/(300*1+300*16))) #单独给一层初始化参数 ，给定方差，
		#后面再有一次weight_init初始化，输入和输出的方差相同，所以也相当于用到这里方差，只是二次初始化有利于收敛
		
		self.conv2=mymodule() #于是Net 就有了两个子module  convE1  myumodule
		
		self.W=nn.Parameter((torch.rand(90,90)), requires_grad=True)# W不是一个module，只是一个tensor容器，跟convE1.weight和convE1.bias是同级的
		print('卷积层weight和bias一起:',self.convE1)
		
		print('卷积层weight',self.convE1.weight)
		print('卷积层bias',self.convE1.bias)
		print('初始化前',self.W)
		nn.init.normal_(self.W.data)
		print('初始化后',self.W)
	
# for I in range(10):  #10折
	# net=Net(16,16,16,64,256) #注意，每折之间，需要重新定义网络，内部已经有初始化，这样才能保证每两折之间相互独立
	# net.apply(weights_init) #重新二次初始化
	# #net = nn.DataParallel(net,device_ids=[0,1,2])
	# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0)
	# for epoch in range(70):
		
#python 使用nn.init 参数初始化
# 参考 http://www.cnblogs.com/lindaxin/p/8037561.html
# https://blog.csdn.net/dss_dssssd/article/details/83959474
# torch.nn.init.constant(tensor, val) #初始化为常数
# torch.nn.init.normal_(tensor, mean=0, std=1) #正态分布
# torch.nn.init.uniform_(tensor, a=0, b=1)     #均匀分布

# torch提供两种xavier 分别是uniform 和normal
#xavier表示输入和输出方差相同，如果初始化值很小，则后面方差会趋于0，接近线性，失去非线性。太大就会导致反向传播的梯度消失 1/参数
# torch.nn.init.xavier_uniform(tensor, gain=1) 
# torch.nn.init.xavier_normal_(tensor, gain=1)

#kaiming
# torch.nn.init.kaiming_uniform_(tensor, a=0, mode=‘fan_in’, nonlinearity=‘leaky_relu’)
# torch.nn.init.kaiming_normal_(tensor, a=0, mode=‘fan_in’, nonlinearity=‘leaky_relu’)


# 1.对某层参数进行初始化
# self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,3), stride=2, padding=3)   #有.weight的属性
# init.xavier_uniform(self.conv1.weight)
# init.constant(self.conv1.bias, 0.1)


# 对整个网络进行参数初始化  通过前缀找module
# def weights_init(m):
	# classname=m.__class__.__name__  #获取子模块的名字
	# if classname.find('conv') != -1: #网络中定义了self.conv这个变量，这里就找变量前缀为conv的
		# nn.init.xavier_uniform_(m.weight.data)
		# nn.init.xavier_uniform_(m.bias.data)
		
#上面是通过前缀找module，这里是根据module的类别来找
def weights_init(m):
	print('进来的是什么:',m)
	if 'weight' in dir(m):
		print('含有weight')  #这个是能判断  mymodule对象.weight存在的
	
	if isinstance(m, nn.Conv2d):
		print('找到conv2d')
		print('初始化前的conv2d权重',m.weight)
		nn.init.normal_(m.weight) #xavier均值为0,方差为(1/n)
		print('初始化后的conv2d权重',m.weight)
		nn.init.normal_(m.bias) #传入tensor就好了   tensor.data也行
	if isinstance(m, nn.Parameter):
		print('找到parameter')
		
net = Net()
print('有哪些模块:',net)#可以net.convE1  输入模块显示括号里面的
print('类型',type(net))
net.apply(weights_init) #递归搜索所有module,如nn.conv2d，,nn.linear   这个模块找不到nn.Parameter。
# 只会输入模块，不输入tensor或者容器，所以nn.Parameter可以封装到一个小module类，然后再写进来会好一些

print('外部初始化前的nn.Parameter非模块:',net.W)
nn.init.normal_(net.W) #可以从外部单独改变net中nn.Parameters的初始化  
#init.normal 只能放入容器,tensor类型，不能放入module    w是容器，convE1是module，底下的weight和bias是两个容器

print('外部初始化后nn.Parameter非模块:',net.W)  # 这里nn.Parameter被初始化了   #net就是一个容器，一个字典


#可以先打印看net有什么模块module，然后net.key去选中模块，注意nn.parameters这里找不到，因为他不是net的module,是net.parameters
#如果net中有sequence，可以先看net有什么key，然后再读相关模块
print('网络.parameters1',net.parameters)
print('网络.parameters2',list(net.parameters()))#能看得到，nn.Parameters在那里面的
print('net字典',net.state_dict())
print('net字典的key',net.state_dict().keys())#参数的字典  有nn.Para
# print(torch.numel(x))#查看tensor的参数个数


# def weights_init(m): #加上这一步初始化更好收敛
	# if isinstance(m, nn.Conv2d):
		# #nn.init.kaiming_normal_(m.weight, mode='fan_out')
		# nn.init.xavier_uniform_(m.weight)
		# nn.init.constant_(m.bias, 0)
	# elif isinstance(m, nn.Linear):
		# nn.init.xavier_uniform_(m.weight)
		# nn.init.constant_(m.bias, 0)
		
for sub_module in net.children():
	print('子模块',sub_module)
	



for name, Parameter in net.named_parameters():#这里是读容器，不是读模块 全部都有  每个装着参数的都是容器  W就是一个普通容器，conv那些是模块，也是容器
    print('查看named_parameters:',name, '参数是',Parameter) #












		
		
		
		
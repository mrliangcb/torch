#参考 https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308

#leaf variable和普通variable
x = torch.autograd.Variable(torch.Tensor([1, 2, 3, 4]))  #是叶子变量 leaf variable
y = x + 1  # 非叶子变量  variable

#还有in-place变量
x += 1  # in-place  仍是自己的，就是in place
y = x + 1 # not in place 新建了一个变量


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
	
	
	
	
	
	
	
	
	
	
	
	
	




















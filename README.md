# pytorch 学习笔记

## 1.[介绍tensor 和 variable](./chapter1.ipynb)  
讲了numpy 和 tensor  variable的关系    
求导问题：variable和 nn.parameter的区别 variable适合手动更新var.data=……     nn.parameter适合用optimizer更新  
叶子变量leaf variable的问题    
统计variable.sum()  .mean()  
维度操作:Gm=Gm.view([1,1,30,30])#可以把[30,30]的矩阵扩展维度   
torch.cat((Gm,Gv),1)#在哪个维度上连接起来,torch中一般是[bath数,深度，高度，宽度]

### 2.[建立一个简单的线性模型](./线性回归模型.py)  
文中建立了variable 的权重  
用y=wx+b去模拟数据,x是一个数,w也是一个数  
手动写的MSE和权重更新 w.data=更新  

### 3.[简单两层神经网络](./)  
模型跟上面相似y=wx+b  
但x是二维的向量 ,w也是二维矩阵shape=(2,1)  
此时用自动的优化策略torch.optim.SGD([要更新的参数,],1.)  或者(model.parameters(),1.)  
自动优化只能更新W=nn.Parameter()类型的参数，不能更新variable类型的。  
因为定义了nn.Parameter,就会绑定到nn.model类中，然后optimizer就会找到这个nn.Parameter进行更新  

- 优化步骤  
- 定义优化器（包含哪些要求导更新的变量）和loss函数  
- 初始化优化器optimizer.zero_grad()  
- 网络输出，计算loss，loss.backward（） 求导    
- optimizer.step()根据求导来更新  

### 4.[把module写入类](./神经网络(结构包在类里).py)  
上面的例子模型写在函数，现在写入一个类,类用的比较广泛  

### 5.[y=w1x+w2x^2+w3x^3+b 多项式模型](./线性多项式回归.py)  
手动更新  

### 6.[CNN做Mnist](./practical)  
平时做的模型收藏在这里[my_model](./practical/my_model.py)  

### 7.[y=wx 与CNN串联  ](./practical/cnn+WX.py)  
> 可以参考成品[用两个optimizor分别优化线性和model.parameters](./practical/experience1_adhd/main4.0.py)
> 之前的联系都是直接调用CNN的接口，没有自定义权重  
> 或者只对简单模型Y=kx+b（神经网络或线性模型）(数据X是一个数的，就是线性模型，如果是二维的，则是神经网络)    
> 现在对两者进行结合，重点是能对自定义的权重进行求导，更新    
> 问题:  可以参考[inplace和free buffer报错如何解决](./test_leaf.py)    
- 1.11x+2=y1  x*2=y2   用一个矩阵表示z=[y1,y2]   (这个矩阵就相当于中间变量)  
  那么要先新建一个function函数,function里面建立z tensor，然后再z[0,0]=   
- 2.注意等号的两边不能有同样的角色，如tensor1=tensor1，但可以tensor1=tensor1.clone(),通常可以新建0tensor矩阵来接受值，这样是可以求导的
 (感觉应该可以，+=就是在原来的基础上加，但a=a+1，左边的a可能是新的a，右边的是旧的，经过上面问题的实验，是可以的)
 
- 3. ``0.4`` 版本之后的torch都不支持全部inplace  
- 4.如何调试求导哪里出错了呢？用以下方法：

```
print('求导')   
variable1=variable.mean()   #对于矩阵型的variable，则要联成一个数才能求导  
variable1.backward(retain_graph=True)  
print('求导成功')   
```  
- 5.a=torch.div(a,2)是会发生inplace错误，需要b=a.clone()一下，然后b=torch.div(a,2)  
  (a数组或者单变量的tensor)a+=a,a=a+2都没有错  


### 8.调试工具
  variable.requires_grad  
  variable.is_leaf  
  
### 9.autograd.backward()里面参数的用法  
参考``https://blog.csdn.net/qq_17550379/article/details/78939046``  
特别是.backward()求导完之后，就会立即释放，不能再次求导了。因此如果想在程序多处求导（调试程序），则要用retain_graph=True  

### 10.小细节经验:  
- (1)做concat，想避免inplace操作，可以在i=0时候新建一个变量,i>0的时候就concat到第0个的结果  
for i in range(len())  

- (2)累加可以 新建一个长为t的数组，将t次结果记录下来，然后求.sum()  

### 11.cuda变量输入到函数中  
```
def pl(a,b):
	c=torch.Tensor(1).cuda(0)
	for i in range(10000):
		c=c+a+b #如果只是c=a+b是没问题的，但c=c+a+b的话，则要把c放到cuda先，这里感觉左边的c是新建的c，cuda变量之间才能运算  
	print('结果c:',c)  
a=Variable(torch.Tensor(1)).cuda(0)  
b=Variable(torch.Tensor(1)).cuda(0)  
pl(a,b)    
```   
### 12.如何把变量放到gpu上？  对tensor或者variable 的末尾加上.cuda(0)  
如何看这个变量是否cuda上的，print一下它，如果最后出现device=cuda就是了  
  
### 13.loss = criterion(outputs, label_variable)  注意输入标签是variable形，output应该还是variable.cuda型，  
这里要注意，label，如果output在cpu上，label_var也要在cpu，或者共同都在cuda上  
而且label_var是Int型的:  
```
label=np.array(y).astype(np.int64)  
loss = criterion(outputs, Variable(torch.tensor(label)))  
```

### 14.分类网络的输出  
> 一般是[[0.3,0.6],[0.2,0.7]]二维矩阵，batch是多少，它就有多少行,每行里面有两个概率对应两个类  

### 15.求准确率的问题
> 可以参考[预测最大类别](./预测最大类别.py)  
 
### 16.variable放上cuda，报错不能求导的问题 non-leaf variable  
'x = Variable(torch.FloatTensor(some_np_array).cuda(), requires_grad=True)'    
以前的写法是先建立一个variable再给.cuda()，但这样相当于a=a.cuda，左边的a是开辟新的a了，    
而不是我们原本建立的,所以可以在建立tensor的时候就可以给cuda，然后再给variable  
如果是不用求导的变量，像data,label的variable则不用这样，可以直接把variable放上cuda  


### 17.模型参数的保存和加载
可以参考[例子1](./保存权重例子.py) , [例子2](./保存读取权重.py)   
注意读取的时候，要先建立一个规模和之前一样的W或者net  




# pytorch 学习笔记

## 1.[介绍tensor 和 variable](./chapter1.ipynb)  
讲了numpy 和 tensor  variable的关系    
求导问题：variable和 nn.parameter的区别 variable适合手动更新var.data=……     nn.parameter适合用optimizer更新  
叶子变量leaf variable的问题    
统计variable.sum()  .mean()


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
之前的联系都是直接调用CNN的接口，没有自定义权重  
或者只对简单模型Y=kx+b（神经网络或线性模型）(数据X是一个数的，就是线性模型，如果是二维的，则是神经网络)  
现在对两者进行结合，重点是能对自定义的权重进行求导，更新  
问题:  可以参考[inplace和free buffer报错如何解决](./test_leaf.py)
- 1.11x+2=y1  x*2=y2   用一个矩阵表示z=[y1,y2]   (这个矩阵就相当于中间变量)
  那么要先新建一个function函数,function里面建立z tensor，然后再z[0,0]=   
- 2.注意等号的两边不能有同样的角色，如tensor1=tensor1，但可以tensor1=tensor1.clone(),通常可以新建0tensor矩阵来接受值，这样是可以求导的
- 3. ``0.4`` 版本之后的torch都不支持全部inplace  
- 4.如何调试求导哪里出错了呢？用以下方法：
```
print('求导')   
variable1=variable.mean()   #对于矩阵型的variable，则要联成一个数才能求导
variable1.backward()  
print('求导成功')   
```  

### 8.调试工具
  variable.requires_grad
  variable.is_leaf
  
### 9.



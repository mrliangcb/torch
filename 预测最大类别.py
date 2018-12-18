#求预测的类别
import torch
import numpy as np
from torch.autograd import Variable




a = np.array([[1,2,3],
				[6,4,5],
				[8,10,9]])
tensor1 = torch.Tensor(a)
x = Variable(tensor1, requires_grad=True).int()
print(x,'\n类型\n',x.numpy())

print('\n取出最大\n',torch.max(x, 1))#按行取出最大的那个值，放在一个一维的valuable，而且也返回最大下标
#(tensor([ 3,  6, 10], dtype=torch.int32), tensor([2, 0, 1])) 返回了一个tuple，每个行最大

#方法一:
#value转tensor，再转np。
pred_y = torch.max(x, 1)[1].data.numpy()#预测最大的
print('\n最大下标\n',pred_y)

#如果在gpu上，要换成cpu计算numpy，基于numpy计算sum
test_pred = torch.max(softmax_outputs, 1)[1].cpu().data.numpy()
ground_true=test_y.cpu().data.numpy()
accuracy=sum(test_pred==ground_true)/len(ground_true)

#直接基于gpu计算,pred和label都是cuda类型的，这种就不用转成cpu因为不用到numpy
_, pred = out.max(1)
num_correct = (pred == label).sum().data[0]
acc = num_correct / im.shape[0]


# 方法二：
# 或者用numpy的argmax，对numpy操作
a = np.array([[3, 1, 2, 4, 6, 1],[1,2,3,4,5,6]])
print('最大下标',np.argmax(a,axis=1))


#计算有多少个正确，sum(np1,np2)
b=np.array([1,2,3])
c=np.array([1,2,3])
d=len(c)
accuracy=np.sum(b==c)/d





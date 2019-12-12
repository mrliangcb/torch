import sys
import os
import torch
import numpy as np

# curr=os.path.dirname(sys.argv[0])
curr=os.path.abspath(sys.argv[0])
curr=os.path.dirname(curr) #当前文件夹路径
print('当前路径',curr)

class Dataset(object):
	"""An abstract class representing a Dataset.
	All other datasets should subclass it. All subclasses should override
	``__len__``, that provides the size of the dataset, and ``__getitem__``,
	supporting integer indexing in range from 0 to len(self) exclusive.
	"""
	def __init__(self, transform=None, update_dataset=False):#这里一开始就获得名单列表
		# dataset_type: ['train', 'test']
		data_path=curr+r'\data.txt'
		with open(curr+r'\data.txt', 'r') as f:
			lines = f.readlines()#获得 路径_类别的资料
		self.sam_list=[]
		for i in lines:
			i=i.strip()
			self.sam_list.append(i)
		
		
	def __getitem__(self, index):
		item = self.sam_list[index]#获得目录
		ddir,label= item.split('_*')
		img=torch.Tensor(np.loadtxt(curr+r'\\'+ddir,delimiter=',',dtype=float))#读数据为矩阵
		
		label=torch.Tensor([int(label)])#斋数字，表示这个容器装多少

		return img,label
		
	def __len__(self):
		return len(self.sam_list)

	def __add__(self, other):
		return ConcatDataset([self, other])

# 创建了dataset之后用迭代器读
# data_set=my_dataset.Dataset()
# import torch.utils.data.dataloader as DataLoader
# data_loader = DataLoader.DataLoader(data_set,batch_size=batch_sizes, shuffle = True)#num_workers=
# for i in data_loader:
	# img,label=i
	# print(img,label)







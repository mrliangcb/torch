#实验保存单个权重

import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import read_data

save_state=False

if save_state:
	W= nn.Parameter((torch.ones(116,116)*0.1), requires_grad=True) #
	state = {'W':W}
	ckpt_path=r'C:\Users\mrliangcb\Desktop\adhd_pro\W.pkl'
	torch.save(state,ckpt_path)
	
else:
	W= Variable((torch.Tensor(116,116)))
	ckpt_path=r'C:\Users\mrliangcb\Desktop\adhd_pro\W.pkl'
	checkpoint = torch.load(ckpt_path,map_location=lambda storage, loc: storage)
	#checkpoint = torch.load(ckpt_path)
	W = checkpoint['W']
	print(W)
	
	
例子:
#保存
	state = {'W':W,'net':net.state_dict(), 'optimizer1':optimizer.state_dict(),'optimizer2':optimizer2.state_dict()}
	ckpt_path=r'./checkpoint/net_params{}.pkl'.format(epoch)
	torch.save(state,ckpt_path)
	
#加载
	# net = net_model.resnet2(1,2)
# ckpt_path=r'./checkpoint/2class_res.clk'
# checkpoint = torch.load(ckpt_path,map_location=lambda storage, loc: storage)
# net.load_state_dict(checkpoint['net'])







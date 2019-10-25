

import torch

opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR,weight_decay=1e-4)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8,) #sgd的优化版
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))  #betas 一阶和二阶的权重衰减率 用于计算梯度以及梯度平方的运行平均值的系数
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]



# 中途修改学习率
optimizer.param_groups[0]['lr'] = 1e-5

#或者多个参数组
for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-1


if torch.cuda.is_available():





















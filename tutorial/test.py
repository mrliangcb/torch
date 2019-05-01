

import torch
from torch.autograd import Variable


a=Variable(torch.Tensor([3]), requires_grad=True)
b=torch.nn.Parameter(torch.Tensor(a.size()), requires_grad=True)

print(type(a),type(b))








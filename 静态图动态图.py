#静态图和动态图
import torch
first_counter = torch.Tensor([0])
second_counter = torch.Tensor([10])
while (first_counter < second_counter)[0]:
    first_counter += 2
    second_counter += 1
print(first_counter)
print(second_counter)

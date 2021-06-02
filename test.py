import torch
import RL_Header as RL

Agent = RL.Agent()

arr = torch.tensor([[3212.,   61.,    0.,    0.,    0.],[4222.,   62.,    0.,    0.,    0.]], device='cuda:0')
print(arr[0][1])

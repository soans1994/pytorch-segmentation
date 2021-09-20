import torch
import torchvision
import numpy
print(torch.__version__)
print(torchvision.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([[1,2,3,4,5,6], [2,3,4,5,6,7]], dtype=torch.float32, device=device, requires_grad=True)
y = torch.tensor([[1,2,3,4,5,6], [2,3,4,5,6,7]],device=device)
x1 = [[1,2,3,4,5,6], [2,3,4,5,6,7]]
y1 = [[1,2,3,4,5,6], [2,3,4,5,6,7]]
z= x+y
z1 = numpy.asarray(x1)+numpy.asarray(y1)
print(z, z.device, z.shape)
print(z[0:2,0:4:3])#tensor also same as numpy(add extra to index)
print(z1, z.shape, z1.dtype)
print(z1[0:2,0:4:3])#numpy(add extra to index)
import torch
import numpy as np
from torch.nn import Linear,Conv2d


class PDE_NET(torch.nn.Module):
    def __init__(self, blocks, symnet, gradient_generator):
        super(PDE_NET, self).__init__()
        self.blocks = blocks
        self.symnet = symnet     
#         self.generate_grads = generate_grads
        self.gradient_generator = gradient_generator
        
    def forward(self,images):
        grads = torch.Tensor(np.stack(self.gradient_generator.generate(images),axis=-1))
        images_preds = []
        for block in range(self.blocks):
            preds = self.symnet(grads)
            images_preds.append(preds)
            grads = torch.Tensor(np.stack(self.gradient_generator.generate(preds.detach().numpy()),axis=-1))
        return torch.stack(images_preds,axis=-1)
    
class LinearSymNet(torch.nn.Module):
    def __init__(self, in_size,k):
        super(LinearSymNet, self).__init__()
        self.k = k
        self.mults = torch.nn.ModuleList([Linear(in_size+i,2,bias=False) for i in range(k)])            
        self.output = Linear(in_size+k,1)
        
    def forward(self,x):
        initial_shape = x.shape
        flattened_x = torch.Tensor(x.view(-1, x.shape[-1]))
        for model in self.mults:
            mult = model(flattened_x)
            flattened_x = torch.cat((flattened_x,(mult[:,0]*mult[:,1]).unsqueeze(-1)),axis=-1)
        output_x = self.output(flattened_x) 
        final_x = output_x.view(initial_shape[:-1],1)
        return final_x
    
class ConvSymNet(torch.nn.Module):
    def __init__(self, in_size,k, kernel_size):
        super(ConvSymNet, self).__init__()
        self.k = k
        self.mults = torch.nn.ModuleList([Linear(in_size+i,2,bias=False) for i in range(k)])            
        self.output = Conv2d(in_size+k,1,kernel_size,padding=kernel_size//2)
        
    def forward(self,x):
        initial_shape = x.shape
        flattened_x = torch.Tensor(x.view(-1, x.shape[-1]))

        for model in self.mults:
            mult = model(flattened_x)
            flattened_x = torch.cat((flattened_x,(mult[:,0]*mult[:,1]).unsqueeze(-1)),axis=-1)
        preconv_x = flattened_x.view(list(initial_shape[:-1]) + [-1])
        output_x = self.output(preconv_x.permute(0,3,1,2)).squeeze()
        return output_x
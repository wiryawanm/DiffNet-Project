from DiffNet import DiffNetLayer
from queue import Queue
import torch
from torch.nn import Linear, Conv2d,ReLU
import numpy as np

class GradientDiffNet(torch.nn.Module):
    """
    The Gradient DiffNet Class
    A DiffNet implementation with image gradients incorporated for modelling diffusion problems

    ...

    Attributes
    ----------
    nlayers : int
        number of diffusion layers in DiffNet
    k : int
        number of CNN layers in the approximator of diffusivity stencil
    layer_size : int
        number of channels in the hidden CNN layers of the diffusivity approximator
    max_degree : int
        the maximum order of image partial derivatives to be included in the network
    """

    def __init__(self,nlayers,k, max_degree=2, layer_size = 32, dt = 0.1):
        super(GradientDiffNet, self).__init__()
        self.max_degree = max_degree
        self.gradientGenerator = GradientGenerator(max_degree)
        self.layers = torch.nn.ModuleList([DiffNetLayer(k,in_channel=self.gradientGenerator.n_variables,out_channel=self.gradientGenerator.n_variables*5) for i in range(nlayers)])
        self.nlayers = nlayers
        self.k = k
        self.dt = dt
        self.name = 'GradientDiffNet'
        self.model_name = '_'.join([self.name,str(nlayers),str(k),str(max_degree)])


    def forward(self,x):
        x_update = x + 0
        for i in range(self.nlayers):
          grads = self.gradientGenerator.generate(x_update.cpu().detach().numpy())
          np_grads = np.concatenate(grads,axis=1)
          x_input = torch.Tensor(np_grads).to(0)
          kappa = self.layers[i](x_input)
          size = x_update.shape
          summation = torch.zeros(x_update.shape).to(0)

          for j, grad in enumerate(grads):
            x_grad = torch.Tensor(grad).to(0)
            xdirections = torch.zeros((size[0],5,size[2],size[3])).to(0)

            xdirections[:,0,:,:] = torch.cat([torch.zeros((size[0],size[1],1,size[3])).to(0), x_grad[:,:,:-1,:]], dim = 2).squeeze()
            xdirections[:,1,:,:] = torch.cat([x_grad[:,:,1:,:], torch.zeros((size[0],size[1],1,size[3])).to(0)], dim = 2).squeeze()
            xdirections[:,2,:,:] = torch.cat([torch.zeros((size[0],size[1],size[2],1)).to(0), x_grad[:,:,:,:-1]], dim = 3).squeeze()
            xdirections[:,3,:,:] = torch.cat([x_grad[:,:,:,1:], torch.zeros((size[0],size[1],size[2],1)).to(0)], dim = 3).squeeze()
            xdirections[:,4,:,:] = (x_grad * -1).squeeze()
                          
            x_gamma = torch.mul(xdirections,kappa[:,j:j+5,:,:])*self.dt 
            x_grad_update = torch.sum(x_gamma,dim=1).unsqueeze(1)
            summation += x_grad_update

          x_update += summation
        return x_update

        

class GradientGenerator:
    """
    The Gradient Generator Class
    A class which handles the generation of image spatial derivatives
    ...

    Attributes
    ----------
    max_degree : int
        the maximum order of image partial derivatives to be included in the network
    """
    def __init__(self,max_degree):
        self.max_degree = max_degree
        self.variables = self.generate_variables()
        self.n_variables = len(self.variables)
        
    def generate(self,images):
        images_grads = [images]
        q = Queue()
        q.put(images)

        for degree in range(self.max_degree):
            for i in range(q.qsize()):
                g = q.get()
                gx,gy = np.gradient(g,axis=(2,3))
                q.put(gx)
                images_grads.append(gx)
            q.put(gy)
            images_grads.append(gy)
        return images_grads
    
    def generate_variables(self):
        variables = ['g']
        
        for i in range(1,self.max_degree+1):
            for j in range(i+1):
                variables.append('g' + 'x'*(i-j)+'y'*j)
        
        return variables



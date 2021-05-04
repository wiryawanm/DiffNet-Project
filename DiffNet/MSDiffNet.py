import torch
from torch.nn import Linear, Conv2d,ReLU,ConvTranspose2d
from DiffNet import DiffNetLayer


class MultiScaleDiffNet(torch.nn.Module):
    """
    The Multi-Scale DiffNet Class
    A DiffNet implementation with muti-scale information incorporated for modelling diffusion problems

    ...

    Attributes
    ----------
    nlayers : int
        number of diffusion layers in DiffNet
    k : int
        number of CNN layers in the approximator of diffusivity stencil
    layer_size : int
        number of channels in the hidden CNN layers of the diffusivity approximator
    scale_level : int
        the maximum level of coarse scale representations to be included in the network
    """
    def __init__(self,nlayers,k, layer_size = 32, dt = 0.1, scale_level = 2):
        super(MultiScaleDiffNet, self).__init__()
        self.nlayers = nlayers
        self.k = k

        layers = [torch.nn.ModuleList([DiffNetLayer(k,in_channel=2) for i in range(nlayers)]) for j in range(scale_level)]
        layers.append(torch.nn.ModuleList([DiffNetLayer(k,in_channel=1) for i in range(nlayers)]))
        self.diffnetlayers = torch.nn.ModuleList(layers)

        self.upconvs = torch.nn.ModuleList([DiffNetUpConv(3) for i in range(scale_level+1)])

        self.dt = dt
        self.scale_level = scale_level
        self.meanpool = torch.nn.AvgPool2d(2,2)
        self.relu = torch.nn.ReLU()
        self.name = 'MultiScaleDiffNet'
        self.model_name = '_'.join([self.name,str(nlayers),str(k),str(scale_level)])

    def forward(self,x):
        result = []
        scales = [x]
        for j in range(self.scale_level):
          scales.append(self.meanpool(scales[-1]))

        prev_x_scale = 0
        for j in range(self.scale_level,-1,-1):
          x_scale = scales[j]
          layers = self.diffnetlayers[j]

          for i in range(self.nlayers):
            if j < self.scale_level:
              x_scale = torch.cat([x_scale,prev_x_upscale],dim=1)

            kappa = layers[i](x_scale)
            xdirections = self.directions(x_scale[:,0:1])
            x_gamma = torch.mul(xdirections,kappa)*self.dt 
            x_scale = torch.sum(x_gamma,dim=1).unsqueeze(1) + x_scale[:,0:1]

          result.append(x_scale)
          prev_x_upscale = self.upconvs[j](x_scale)
          
        return result[-1]

    def directions(self,x):
        size = x.shape
        xdirections = torch.zeros((size[0],5,size[2],size[3])).to(0)
        
        xUp = torch.cat([torch.zeros((size[0],size[1],1,size[3])).to(0), x[:,:,:-1,:]], dim = 2).squeeze()
        xDown = torch.cat([x[:,:,1:,:], torch.zeros((size[0],size[1],1,size[3])).to(0)], dim = 2).squeeze()
        xLeft = torch.cat([torch.zeros((size[0],size[1],size[2],1)).to(0), x[:,:,:,:-1]], dim = 3).squeeze()
        xRight = torch.cat([x[:,:,:,1:], torch.zeros((size[0],size[1],size[2],1)).to(0)], dim = 3).squeeze()

        xdirections[:,0,:,:] = xUp
        xdirections[:,1,:,:] = xDown
        xdirections[:,2,:,:] = xLeft
        xdirections[:,3,:,:] = xRight
        xdirections[:,4,:,:] = (x * -1).squeeze()

        return xdirections

        
class DiffNetUpConv(torch.nn.Module):
    """
    The DiffNet Upconvolution Class
    This class handles the upconvolution module of the Multi-Scale DiffNet implementation, consisting of an up-convolution layer and k-layers of CNN

    ...

    Attributes
    ----------
    k : int
        number of CNN layers following the upconvolution step
    layer_size : int
        number of channels in the hidden CNN layers 
    """
    def __init__(self,k, layer_size = 32,in_channel = 1,out_channel=1,kernel_size=3):
        super(DiffNetUpConv, self).__init__()
        padding = kernel_size//2
        if k == 0:
          layers = [ConvTranspose2d(in_channel,out_channel,2,2)]
        elif k > 0:
          layers = [ConvTranspose2d(in_channel,layer_size,2,2)]
          layers += [Conv2d(layer_size,layer_size,kernel_size,padding=padding) for i in range(k-1)]
          layers += [Conv2d(layer_size,out_channel,kernel_size,padding=padding)]  
           
        self.layers = torch.nn.ModuleList(layers)
        self.relu = ReLU()

    def forward(self,images):
        output = images
        for layer in self.layers[:-1]:
            output = self.relu(layer(output))
        output = self.layers[-1](output)
        return output
        


from torch.nn import Linear, Conv2d,ReLU
import torch

class DiffNetLayer(torch.nn.Module):
    def __init__(self,k, layer_size = 32,in_channel = 1,out_channel=5,kernel_size=3):
        super(DiffNetLayer, self).__init__()
        padding = kernel_size//2
        layers = [Conv2d(in_channel,layer_size,kernel_size,padding=padding)]
        layers += [Conv2d(layer_size,layer_size,kernel_size,padding=padding) for i in range(k)]
        layers += [Conv2d(layer_size,out_channel,kernel_size,padding=padding)]   
        self.layers = torch.nn.ModuleList(layers)
        self.relu = ReLU()

    def forward(self,images):
        output = images
        for layer in self.layers[:-1]:
            output = self.relu(layer(output))
        output = self.layers[-1](output)
        return output
    
class DiffNet(torch.nn.Module):
    def __init__(self,nlayers,k, layer_size = 32, dt = 0.1,kernel_size=3):
        super(DiffNet, self).__init__()
        self.diffnetlayers = torch.nn.ModuleList([DiffNetLayer(k,layer_size,kernel_size=kernel_size) for i in range(nlayers)])
        self.dt = dt
        self.name = 'DiffNet'
        self.model_name = '_'.join([self.name,str(nlayers),str(k)])
    def forward(self,x):
        x_update = x
        for layer in self.diffnetlayers:
            kappa = layer(x_update)
            size = x_update.shape
            xdirections = torch.zeros(kappa.shape).to(0)
            
            xdirections[:,0,:,:] = torch.cat([torch.zeros((size[0],size[1],1,size[3])).to(0), x_update[:,:,:-1,:]], dim = 2).squeeze()
            xdirections[:,1,:,:] = torch.cat([x_update[:,:,1:,:], torch.zeros((size[0],size[1],1,size[3])).to(0)], dim = 2).squeeze()
            xdirections[:,2,:,:] = torch.cat([torch.zeros((size[0],size[1],size[2],1)).to(0), x_update[:,:,:,:-1]], dim = 3).squeeze()
            xdirections[:,3,:,:] = torch.cat([x_update[:,:,:,1:], torch.zeros((size[0],size[1],size[2],1)).to(0)], dim = 3).squeeze()
            xdirections[:,4,:,:] = (x_update * -1).squeeze()
                         
            x_gamma = torch.mul(xdirections,kappa)*self.dt 
            x_update = torch.sum(x_gamma,dim=1).unsqueeze(1) + x_update
            
        return x_update
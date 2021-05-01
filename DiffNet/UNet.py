import torch
from torch.nn import Linear, Conv2d,ReLU,ConvTranspose2d

class UNet_CNN(torch.nn.Module):
    def __init__(self,k, layer_size = 32,in_channel = 1,kernel_size=3):
        super(UNet_CNN, self).__init__()
        padding = kernel_size//2
        layers = [Conv2d(in_channel,layer_size,kernel_size,padding=padding)]
        for i in range(k):
          layers.append(Conv2d(layer_size,layer_size,kernel_size,padding=padding))
        self.layers = torch.nn.ModuleList(layers)
        self.relu = ReLU()

    def forward(self,images):
        output = images
        for layer in self.layers[:-1]:
            output = self.relu(layer(output))
        output = self.layers[-1](output)
        return output


class UNet(torch.nn.Module):
    def __init__(self,k, layer_size = 32, dt = 0.1, scale_level = 2,kernel_size=3):
        super(UNet, self).__init__()

        curr_layer_size = layer_size
        curr_in_channel = 1

        down_cnns = []
        for i in range(scale_level+1):
          down_cnns.append(UNet_CNN(k,curr_layer_size, curr_in_channel,kernel_size=kernel_size))
          curr_in_channel = curr_layer_size
          curr_layer_size *= 2
        
        curr_layer_size = int(curr_in_channel/2)

        up_cnns = []
        up_convs = []
        for i in range(scale_level):
          up_cnns.append(UNet_CNN(k,curr_layer_size, curr_in_channel))
          up_convs.append(torch.nn.ConvTranspose2d(curr_in_channel,curr_layer_size,2,2))
          curr_in_channel = curr_layer_size
          curr_layer_size = int(curr_layer_size/2)

        
        self.down_cnns = torch.nn.ModuleList(down_cnns)
        self.up_cnns = torch.nn.ModuleList(up_cnns)
        self.maxpool = torch.nn.MaxPool2d(2,2)
        self.up_convs = torch.nn.ModuleList(up_convs)
        self.relu = torch.nn.ReLU()

        self.final_conv = torch.nn.Conv2d(layer_size,1,kernel_size,padding = kernel_size//2)

        self.dt = dt
        self.scale_level = scale_level

        self.name = 'U-Net'
        self.model_name = '_'.join([self.name,str(k),str(scale_level)])

    def forward(self,images):
        x = images

        down_outputs = []
        for i,down_cnn in enumerate(self.down_cnns):
          x = down_cnn(x)
          if i < self.scale_level: 
            down_outputs.append(x)
            x = self.maxpool(x)

        down_outputs.reverse()

        for i, up_cnn in enumerate(self.up_cnns):
          up_conv = self.up_convs[i]
          x = torch.cat([down_outputs[i], self.relu(up_conv(x))],dim=1)
          x = up_cnn(x)
        
        x = self.relu(images + self.final_conv(x))
        # print(x.shape)

        return x

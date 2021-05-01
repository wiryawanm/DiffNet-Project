import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy
from queue import Queue
from PDENet import PDE_NET


def train_PDE_NET(symnet,gradient_generator, blocks, trainData, validData, batch_size, lr, max_epochs, early_stopping=10):
    trainInputs = torch.Tensor(trainData[:,:,:,0])
    trainSteps = torch.Tensor(trainData[:,:,:,1:1+blocks])
    trainImagesDataset = TensorDataset(trainInputs,trainSteps)
    trainImagesLoader = DataLoader(trainImagesDataset, batch_size=batch_size, shuffle = True)

    validInputs = torch.Tensor(validData[:,:,:,0])
    validSteps = torch.Tensor(validData[:,:,:,1:1+blocks])

    pdenet = PDE_NET(blocks = blocks, symnet = symnet,gradient_generator = gradient_generator)
    optimizer = torch.optim.AdamW(pdenet.parameters(), lr=lr, weight_decay=0.05)
    criterion = torch.nn.MSELoss()

    
    train_loss = criterion(pdenet(trainInputs),trainSteps).item()
    valid_loss = criterion(pdenet(validInputs),validSteps).item()
    print('Epoch: ', 0,end='\t')
    print('MSE train loss: ', train_loss,end='\t')
    print('MSE valid loss: ', valid_loss)  
    
    min_valid_loss = valid_loss
    convergence_counter = 0
    best_model = pdenet
    for epoch in range(1,max_epochs+1):
        for xs,ys in trainImagesLoader:
            pdenet.zero_grad()
            y_preds = pdenet(xs)
            batch_loss = criterion(y_preds,ys)
            batch_loss.backward()
            optimizer.step()

        train_loss = criterion(pdenet(trainInputs),trainSteps).item()
        valid_loss = criterion(pdenet(validInputs),validSteps).item()

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            convergence_counter = 0
            best_model = copy.deepcopy(pdenet)
        else:
            convergence_counter += 1
            if convergence_counter >= early_stopping: break

        print('Epoch: ', epoch, end='\t')
        print('MSE train loss: ', train_loss,end='\t')
        print('MSE valid loss: ', valid_loss)

    print('Epoch: ', epoch,end='\t')
    print('MSE train loss: ', train_loss,end='\t')
    print('MSE valid loss: ', valid_loss)  
    
    return best_model



class GradientGenerator:
    def __init__(self,max_degree):
        self.max_degree = max_degree
        self.variables = self.generate_variables()
        self.n_variables = len(self.variables)
        
    def generate(self,images):
        images_grads = [images]
        q = Queue()
        q.put(images)
#         print('images shape: ' + str(images.shape))
        for degree in range(self.max_degree):
            for i in range(q.qsize()):
                g = q.get()
                gx,gy = np.gradient(np.pad(g,[(0,0),(1,1),(1,1)]),axis=(1,2))
                gx = gx[:,1:-1,1:-1]
                gy = gy[:,1:-1,1:-1]
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
    


def display_results(pdenet_model, data, idx,reverse=False):
    blocks = pdenet_model.blocks
    figs, axs = plt.subplots(2,blocks+1, figsize=(20,8))

    dispInputs = torch.Tensor(data[:,:,:,0])
    dispSteps = torch.Tensor(data[:,:,:,1:1+blocks])

    preds = pdenet_model(dispInputs).detach().numpy()[idx]
    truth = dispSteps[idx]
    
    if not reverse:
        axs[0][0].imshow(dispInputs[idx].T,cmap='gray', vmin=0, vmax=1)
        axs[0][0].set_title('Input: t = 0')    
        axs[0][0].axis('off')   

        axs[1][0].imshow(dispInputs[idx].T,cmap='gray', vmin=0, vmax=1)
        axs[1][0].set_title('Ground Truth: t = 0')    
        axs[1][0].axis('off')   


        for i in range(preds.shape[-1]):    
            axs[0][i+1].imshow(preds[:,:,i].T,cmap='gray', vmin=0, vmax=1)
            axs[0][i+1].set_title('Output: t = {}'.format(i+1))    
            axs[0][i+1].axis('off')
        for i in range(truth.shape[-1]):    
            axs[1][i+1].imshow(truth[:,:,i].T,cmap='gray', vmin=0, vmax=1)
            axs[1][i+1].set_title('Ground Truth: t = {}'.format(i+1))
            axs[1][i+1].axis('off')
    
    else:
        axs[0][-1].imshow(dispInputs[idx].T,cmap='gray', vmin=0, vmax=1)
        axs[0][-1].set_title('Input: t = {}'.format(blocks))    
        axs[0][-1].axis('off')   

        axs[1][-1].imshow(dispInputs[idx].T,cmap='gray', vmin=0, vmax=1)
        axs[1][-1].set_title('Ground Truth: t = {}'.format(blocks))    
        axs[1][-1].axis('off')   


        for i in range(preds.shape[-1]):    
            axs[0][blocks-i-1].imshow(preds[:,:,i].T,cmap='gray', vmin=0, vmax=1)
            axs[0][blocks-i-1].set_title('Output: t = {}'.format(blocks-i-1))    
            axs[0][blocks-i-1].axis('off')
        for i in range(truth.shape[-1]):    
            axs[1][blocks-i-1].imshow(truth[:,:,i].T,cmap='gray', vmin=0, vmax=1)
            axs[1][blocks-i-1].set_title('Ground Truth: t = {}'.format(blocks-i-1))
            axs[1][blocks-i-1].axis('off')        
        
    figs.subplots_adjust(wspace=0.05, hspace=0.05)
    figs.tight_layout()
    return figs
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import copy

def train_diffnet(model, trainset, validset, batch_size, lr, early_stopping = 20,verbose = False,max_epochs=2000):
  """
  The main training function for DiffNet and U-Net models.

  Parameters
  ----------
      model : torch.nn.module
          An instance of a DiffNet/GradientDiffNet/MultiScaleDiffNet/UNet object
      trainset : TensorDataset
          training set of a given forward/inverse flow problem
      validset : TensorDataset
          validation set of a given forward/inverse flow problem
      batch_size : int
          batch size for training
      lr : float
          learning rate for Adam optimizer
      early_stopping : int
          the number of epochs the training should continue for after it has observed the last improvement in validation loss, to ensure convergence
      verbose : bool
          set True to print loss at every epoch, otherwise it would print loss only after convergence
      max_epochs: int
          the maximum number of epochs to train for
  """
  batch_size = 10
  trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle = True)
  validLoader = DataLoader(validset, batch_size=5, shuffle = True)

  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  min_valid_loss = float('inf')

  train_loss = 0
  valid_loss = 0
  for xs,ys in trainLoader:
    train_loss += criterion(model(xs),ys).item()/len(trainLoader)
  for xs,ys in validLoader:
    valid_loss += criterion(model(xs),ys).item()/len(validLoader)  

  if verbose:
    print('Epoch: ', 0, end='\t')
    print('MSE train loss: ', train_loss,end='\t')
    print('MSE valid loss: ', valid_loss)
  else:
    print('Epoch: ', 0, end=' ')


  train_losses = []
  valid_losses = []
  for epoch in range(1,max_epochs+1):
      for xs,ys in trainLoader:
          model.zero_grad()
          y_preds = model(xs)
          batch_loss = criterion(y_preds,ys)
          batch_loss.backward()
          optimizer.step()

      train_loss = 0  
      valid_loss = 0
      for xs,ys in trainLoader:
        train_loss += criterion(model(xs),ys).item()/len(trainLoader)
      for xs,ys in validLoader:
        valid_loss += criterion(model(xs),ys).item()/len(validLoader)  

      if verbose:
        print('Epoch: ', epoch, end='\t')
        print('MSE train loss: ', train_loss,end='\t')
        print('MSE valid loss: ', valid_loss,end='\t')
      else:
        print(epoch, end = ' ')

      train_losses.append(train_loss)
      valid_losses.append(valid_loss)

      if valid_loss < min_valid_loss:
          min_valid_loss = valid_loss
          convergence_counter = 0
          best_model = copy.deepcopy(model)
          if verbose:
            print('(best model!)',end='\t')
          else:
            print('(!)', end = ' ')
      else:
          convergence_counter += 1
          if convergence_counter >= early_stopping: break
      
      if verbose:
        print()
  
  train_loss = 0  
  valid_loss = 0
  for xs,ys in trainLoader:
    train_loss += criterion(best_model(xs),ys).item()/len(trainLoader)
  for xs,ys in validLoader:
    valid_loss += criterion(best_model(xs),ys).item()/len(validLoader)  
  print()
  print('Best train loss: ', train_loss,end='\t')
  print('Best valid loss: ', valid_loss,end='\t')

  return best_model,(train_losses,valid_losses)


def display_examples(model, dataset, idxs,figsize = (12,12)):
  """
  A function to display its forward operation given some dataset.

  Parameters
  ----------
      model : torch.nn.module
          An instance of a trained DiffNet/GradientDiffNet/MultiScaleDiffNet/UNet object
      dataset : TensorDataset
          a dataset containing input and output images (ideally test or validation)
      idxs : list
          a list of integer indexes to sample from the dataset to display
  """
  loader = DataLoader(dataset, batch_size=len(dataset))
  for data in loader: xs,ys = data

  fig, axs = plt.subplots(4,len(idxs),figsize = figsize)

  for i in range(len(axs)):
    for j in range(len(axs[i])):
      axs[i][j].axis('off')

  for ax, row in zip(axs[:,0], ['Input','Output','Truth','Difference']):
      ax.annotate(row, xy=(0, 0.5), xytext=(5, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  size='large', ha='left', va='center')

  for i,idx in enumerate(idxs):
    input = xs[idx,0].cpu().detach().numpy()
    output = model(xs[idx:idx +1])[0,0].cpu().detach().numpy()
    ground_truth = ys[idx,0].cpu().detach().numpy()
    difference = ground_truth-output

    axs[0][i].imshow(input,cmap='gray',vmin=0,vmax=1)
    axs[1][i].imshow(output,cmap='gray',vmin=0,vmax=1)
    axs[2][i].imshow(ground_truth,cmap='gray',vmin=0,vmax=1)
    axs[3][i].imshow(difference,cmap='gray')

  fig.tight_layout()
  
def psnr(model, dataset):
  criterion = torch.nn.MSELoss()
  loader = DataLoader(dataset, batch_size=len(dataset))
  loss = 0
  
  for xs,ys in loader:
    loss += criterion(model(xs),ys).item()/len(loader)  
  
  return 10*np.log10(1/loss)


def evaluate_ssim(model, dataset):
  loader = DataLoader(dataset, batch_size=1)
  total = 0
  
  for xs,ys in loader:
    x = model(xs[:,:]).detach().cpu().numpy()[0,0]
    y = ys.detach().cpu().numpy()[0,0]
    total += ssim(x,y,data_range=1)/len(loader)
  
  return total

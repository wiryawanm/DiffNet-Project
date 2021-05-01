import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import h5py

# file = h5py.File('/content/drive/MyDrive/Colab Notebooks/Thesis/stl10_asf.mat','r')
# inData = np.array(file.get('stl10_asf'))  
# steps = 4

# file = h5py.File('/content/drive/MyDrive/Colab Notebooks/Thesis/stl10_warp.mat','r')
# inData = np.array(file.get('warpedImages'))  
# steps = 6

def load(path, dataype, trainsize=400, validsize=50,testsize=50):
    inData = np.load(path+f'stl10_dataset_{dataype}.npy')
    
    trainData = inData[:trainsize]
    validData = inData[trainsize:trainsize+validsize]
    testData = inData[trainsize+validsize:trainsize+validsize+testsize]


    trainxs = torch.tensor(trainData[:,0,:,:]).unsqueeze(1).float().to(0)
    trainys = torch.tensor(trainData[:,-1,:,:]).unsqueeze(1).float().to(0)

    validxs = torch.tensor(validData[:,0,:,:]).unsqueeze(1).float().to(0)
    validys = torch.tensor(validData[:,-1,:,:]).unsqueeze(1).float().to(0)

    testxs = torch.tensor(testData[:,0,:,:]).unsqueeze(1).float().to(0)
    testys = torch.tensor(testData[:,-1,:,:]).unsqueeze(1).float().to(0)

    trainDataset_forward = TensorDataset(trainxs,trainys)
    validDataset_forward = TensorDataset(validxs,validys)
    testDataset_forward = TensorDataset(testxs,testys)

    trainDataset_inverse = TensorDataset(trainys,trainxs)
    validDataset_inverse = TensorDataset(validys,validxs)
    testDataset_inverse = TensorDataset(testys,testxs)

    dataset = {
        'forward': {
            'train': trainDataset_forward,
            'valid': validDataset_forward,
            'test': testDataset_forward
        },
        'inverse': {
            'train': trainDataset_inverse,
            'valid': validDataset_inverse,
            'test': testDataset_inverse
        }
    }

    return dataset


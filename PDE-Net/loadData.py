import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import h5py


def load(path, dataype, trainsize=400, validsize=50,testsize=50):
    inData = np.load(path+f'stl10_pdenet_dataset_{dataype}.npy')
    
    trainData = np.moveaxis(inData[:trainsize],1,-1)
    validData = np.moveaxis(inData[trainsize:trainsize+validsize],1,-1)
    testData = np.moveaxis(inData[trainsize+validsize:trainsize+validsize+testsize],1,-1)

    trainData_inv = np.flip(trainData[:,:,:,0:5],3).copy()
    validData_inv = np.flip(validData[:,:,:,0:5],3).copy()
    testData_inv = np.flip(validData[:,:,:,0:5],3).copy()

    dataset = {
        'forward': {
            'train': trainData,
            'valid': validData,
            'test': testData
        },
        'inverse': {
            'train': trainData_inv,
            'valid': validData_inv,
            'test': testData_inv
        }
    }

    return dataset
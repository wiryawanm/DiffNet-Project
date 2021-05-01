import torch

from helper import train_diffnet, display_examples
from DiffNet import DiffNet
from MSDiffNet import MultiScaleDiffNet
from GradDiffNet import GradientDiffNet
from UNet import UNet
from loadData import load
import numpy as np

model_path = 'trained models/'
data_path = 'datasets/'
torch.cuda.set_device(0)

if __name__ == "__main__":
    flow = 'aniso' # 'aniso', 'asf', or 'warp'
    problem = 'inverse' # 'forward' or 'inverse'

    dataset = load(data_path,flow)

    model = DiffNet(nlayers=3,k=4).to(0)

    best_model, losses_model = train_diffnet(model=model, 
        trainset=dataset[problem]['train'], 
        validset=dataset[problem]['valid'],
        batch_size=10,
        lr=0.001,
        early_stopping=50,
        verbose=True)

    best_model, losses_model = train_diffnet(model=best_model, 
        trainset=dataset[problem]['train'], 
        validset=dataset[problem]['valid'],
        batch_size=10,
        lr=0.0001,
        early_stopping=50,
        verbose=True)

    torch.save(best_model.state_dict(), model_path + f'{best_model.model_name}_{flow}_{problem}')

    idxs = np.random.randint(0,50,6)
    display_examples(best_model, dataset[problem]['valid'],idxs,figsize=(20,10))
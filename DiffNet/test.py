import torch
from DiffNet import DiffNet
from MSDiffNet import MultiScaleDiffNet
from GradDiffNet import GradientDiffNet
from UNet import UNet
from loadData import load
from helper import display_examples, psnr,evaluate_ssim
import numpy as np

model_path = 'trained models/'
data_path = 'datasets/'
torch.cuda.set_device(0)

if __name__ == "__main__":
    flow = 'aniso' # 'aniso', 'asf', or 'warp'
    problem = 'inverse' # 'forward' or 'inverse'

    dataset = load(data_path,flow)
    
    # An instance of DiffNet/GradientDiffNet/MultiScaleDiffNet/UNet.
    # Note that the attributes of this class should match with the trained model saved locally.
    model = DiffNet(3,4).to(0)
    model.load_state_dict(torch.load(model_path + f'{model.model_name}_{flow}_{problem}'))

    # Display random examples from the test dataset
    idxs = np.random.randint(0,50,6)
    display_examples(model, dataset[problem]['test'],idxs,figsize=(20,10))

    # Compute mean PSNR and SSIM of the test dataset
    print('mean PSNR '+ psnr(model,dataset[problem]['test']))
    print('mean SSIM '+ evaluate_ssim(model,dataset[problem]['test']))
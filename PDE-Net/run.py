from helper import GradientGenerator, train_PDE_NET, display_results
from PDENet import LinearSymNet
from EquationGenerator import generate_equation
from loadData import load
import numpy as np

data_path = 'datasets/'

if __name__ == "__main__":
    diffusion = 'isotropic'  # 'isotropic' or 'anisotropic'
    problem = 'forward' # 'forward' or 'inverse'

    dataset = load(data_path,diffusion)

    gradient_generator = GradientGenerator(3)
    symnet = LinearSymNet(in_size = gradient_generator.n_variables,k = 2)  
    pdenet = train_PDE_NET(symnet=symnet, 
                gradient_generator = gradient_generator,
                blocks=4, 
                trainData=dataset[problem]['train'], 
                validData=dataset[problem]['valid'], 
                batch_size=20, 
                lr=0.005,
                max_epochs = 200)

    idx = np.random.randint(0,50)
    fig_iso = display_results(pdenet,dataset[problem]['test'],idx)
    eqn = generate_equation(pdenet)
    print(eqn.show_topk(10))
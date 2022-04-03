import torch
import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from vit import ViT
from nad_computation import GradientCovarianceAnisotropyFinder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def model_gen_fun():
    model = ViT(image_size=32,
                patch_size=4,
                num_classes=1, 
                dim=128,
                depth=3,
                heads=8,
                mlp_dim=256,
                channels=1).eval()
    return model


anisotropy_finder = GradientCovarianceAnisotropyFinder(model_gen_fun=model_gen_fun,
                                                       scale=100,
                                                       num_networks=10000,
                                                       k=1024,
                                                       eval_point=torch.randn([1, 32, 32], device=DEVICE),
                                                       device=DEVICE,
                                                       batch_size=None)

eigenvalues, NADs = anisotropy_finder.estimate_NADs()

np.save('../NADs/ViT_NADs.npy', NADs)
np.save('../NADs/ViT_eigenvals.npy', eigenvalues)

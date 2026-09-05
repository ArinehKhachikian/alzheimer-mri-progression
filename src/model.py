import torch
from monai.networks.nets import DenseNet121, ViT

def get_densenet():
    model = DenseNet121(
        spatial_dims = 3,              # volumetric (1, 96, 112, 96)
        in_channels = 1,               # grayscale MRI (1, 96, 112, 96)
        out_channels = 3               # CN, MCI, Dementia
    )
    return model

def get_vit():
    model = ViT(
        in_channels = 1,                 
        img_size = (96, 112, 96),       # dim. of input image
        patch_size = (16, 16, 16),      # dim. of patch size
        hidden_size = 768,              # dim. of hidden layer
        mlp_dim = 3072,                 # dim. of feedforward layer
        num_layers = 12,                # # of transformer blocks
        num_heads = 12,                 # # of attention heads
        num_classes = 3,                # # of classes if classification is used
        classification = True           # Bool, determines if classification is used
    )
    return model
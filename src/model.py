import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, ViT
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from resnet import resnet10

def get_densenet():
    model = DenseNet121(
        spatial_dims = 3,               # volumetric (1, 96, 112, 96)
        in_channels = 1,                # grayscale MRI (1, 96, 112, 96)
        out_channels = 3                # CN, MCI, Dementia
    )
    return model

def get_vit():
    model = ViT(
        in_channels = 1,                 
        img_size = (128, 128, 128),     # dim. of input image
        patch_size = (16, 16, 16),      # dim. of patch size
        hidden_size = 768,              # dim. of hidden layer
        mlp_dim = 3072,                 # dim. of feedforward layer
        num_layers = 12,                # # of transformer blocks
        num_heads = 12,                 # # of attention heads
        num_classes = 3,                # # of classes if classification is used
        classification = True           # Bool, determines if classification is used
    )
    return model

def get_medicalnet(pretrained_path, num_classes=3, device='cpu'):
    # Load ResNet10 architecture
    model = resnet10(
        sample_input_W = 128,
        sample_input_H = 128,
        sample_input_D = 128,
        shortcut_type = 'B',
        no_cuda=(device == 'cpu'),
        num_seg_classes = num_classes
    )

    # Load pretrained weights
    checkpoint = torch.load(pretrained_path, map_location = device)
    state_dict = checkpoint['state_dict']

    # Remove 'module.' prefix if present (from DataParallel training)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    # Load weights with strict=False to allow mismatched final layer
    model.load_state_dict(new_state_dict, strict=False)

    # Replace final segmentation layer with classification layer
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )

    return model
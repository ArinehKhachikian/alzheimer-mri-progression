from monai.transforms import (
    Compose,
    NormalizeIntensity,
    Resize,
    RandFlip,
    RandRotate90,
    RandGaussianNoise,
    ToTensor
)

def get_train_transforms():
    return Compose([
        NormalizeIntensity(nonzero=True, channel_wise=True),
        Resize(spatial_size=(96,112,96), mode='trilinear'),
        RandFlip(prob=0.5, spatial_axis=None),
        RandRotate90(prob=0.5, max_k=3, spatial_axes=(0,1)),
        RandGaussianNoise(prob=0.5, mean=0, std=0.1),
        ToTensor()
    ])

def get_val_transforms():
    return Compose([
        NormalizeIntensity(nonzero=True, channel_wise=True),
        Resize(spatial_size=(96,112,96), mode='trilinear'),
        ToTensor()
    ]) 
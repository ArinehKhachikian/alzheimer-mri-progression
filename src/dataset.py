import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset

class ADNIDataset(Dataset):
    def __init__(self, master_labels, transform=None):
        # runs once you create the dataset
        # store your dataset here
        self.data = master_labels
        self.transform = transform

    def __len__(self):
        # returns how many samples you have
        return len(self.data)

    def __getitem__(self, idx):
        # given an index number, return one sample
        # this is where you load the NIfTI file and return a tensor + label
        row = self.data.iloc[idx]
        filepath = row['filepath']
        label = int(row['label'])
        img = nib.load(filepath)
        volume = img.get_fdata(dtype=np.float32)

        # Reorder to standard orientation using nibabel
        import nibabel.orientations as nio                                # import nibabel's orientation tools
        orig_ornt = nio.io_orientation(img.affine)                        # read current orientation from the affine matrix
        targ_ornt = nio.axcodes2ornt('RAS')                               # define target orientation as RAS standard
        transform_ornt = nio.ornt_transform(orig_ornt, targ_ornt)         # calculate what swaps/flips are needed
        volume = nio.apply_orientation(volume, transform_ornt)            # apply the transformation to the numpy array

        volume = np.expand_dims(volume, axis=0)

        if self.transform:
            volume = self.transform(volume)
        else:
            volume = torch.FloatTensor(volume)
        return volume, label

class MultimodalDataset(Dataset):
    def __init__(self, dataframe, tensors_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.tensors_dir = tensors_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        ptid = row['PTID']
        label = int(row['label'])

        # Load preprocessed tensor
        tensor_path = os.path.join(self.tensors_dir, f'{ptid}.pt')
        sample = torch.load(tensor_path, weights_only=False)
        volume = sample['volume']

        # Clinical features: MMSE, CDRSB, age, sex, education
        mmse = float(row['MMSCORE']) if pd.notna(row['MMSCORE']) else 0.0
        cdr = float(row['CDRSB']) if pd.notna(row['CDRSB']) else 0.0
        age = float(row['AGE']) if 'AGE' in row and pd.notna(row['AGE']) else 0.0
        sex = 1.0 if row['PTGENDER'] == 'Male' else 0.0
        edu = float(row['PTEDUCAT']) if pd.notna(row['PTEDUCAT']) else 0.0

        clinical = torch.tensor([mmse, cdr, age, sex, edu], dtype=torch.float32)

        return volume, clinical, label
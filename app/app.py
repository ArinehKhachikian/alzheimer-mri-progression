import torch
import nibabel as nib
import numpy as np
import gradio as gr

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import model as model_module
import transforms

PRETRAINED_PATH = "resnet_10_23dataset.pth"
CHECKPOINT_PATH = "best_model_final.pt"

device = torch.device('cpu')    # Hugging face only offers free CPU usage

model = model_module.get_medicalnet(PRETRAINED_PATH, num_classes=3, device='cpu')
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False))
model.eval()

def predict(nifti_file):
    # Load NIfTI file
    img = nib.load(nifti_file.name)
    volume = img.get_fdata(dtype=np.float32)

    # Orientation standardization
    import nibabel.orientations as nio
    orig_ornt = nio.io_orientation(img.affine)
    targ_ornt = nio.axcodes2ornt('RAS')
    transform_ornt = nio.ornt_transform(orig_ornt, targ_ornt)
    volume = nio.apply_orientation(volume, transform_ornt)

    # Add channel dimension
    volume = np.expand_dims(volume, axis=0)

    # Apply val transforms
    transform = transforms.get_val_transforms()
    volume = transform(volume)
    
    # Add batch dimension and run inference
    volume = volume.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(volume)
        probs = torch.softmax(outputs, dim=1)[0]
    
    return {
        'Cognitively Normal (CN)': float(probs[0]),
        'Mild Cognitive Impairment (MCI)': float(probs[1]),
        'Alzheimer\'s Disease': float(probs[2])
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload NIfTI MRI (.nii or .nii.gz)"),
    outputs=gr.Label(label="Diagnosis Probabilities", num_top_classes=3),
    title="Alzheimer's Disease MRI Classifier",
    description="Upload a T1-weighted brain MRI scan in NIfTI format to classify cognitive status. Model: MedicalNet ResNet10 trained on ADNI dataset (559 subjects).",
    examples=None,
)

if __name__ == "__main__":
    demo.launch()
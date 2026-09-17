import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import gradio as gr
import sys
import os

sys.path.append(os.path.dirname(__file__))

import model as model_module
import transforms

PRETRAINED_PATH = "resnet_10_23dataset.pth"
CHECKPOINT_PATH = "best_model_multimodal.pt"

device = torch.device('cpu')

model = model_module.get_medicalnet_multimodal(PRETRAINED_PATH, num_clinical_features=5, num_classes=2, device='cpu')
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False))
model.eval()

def predict(nifti_file, mmse, cdr, age, sex, education):
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

    # Clinical features
    sex_encoded = 1.0 if sex == 'Male' else 0.0
    clinical = torch.tensor([mmse, cdr, age, sex_encoded, education], dtype=torch.float32).unsqueeze(0)

    # Add batch dimension and run inference
    volume = volume.unsqueeze(0).to(device)
    clinical = clinical.to(device)

    with torch.no_grad():
        outputs = model(volume, clinical)
        probs = torch.softmax(outputs, dim=1)[0]

    return {
        'Cognitively Normal (CN)': float(probs[0]),
        'Alzheimer\'s Disease': float(probs[1])
    }

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.File(label="Upload NIfTI MRI (.nii or .nii.gz)"),
        gr.Slider(minimum=0, maximum=30, value=28, step=1, label="MMSE Score (0-30, higher = better)"),
        gr.Slider(minimum=0, maximum=18, value=0, step=0.5, label="CDR Sum of Boxes (0 = normal)"),
        gr.Slider(minimum=50, maximum=95, value=72, step=1, label="Age"),
        gr.Radio(choices=["Male", "Female"], value="Male", label="Sex"),
        gr.Slider(minimum=0, maximum=20, value=16, step=1, label="Years of Education"),
    ],
    outputs=gr.Label(label="Diagnosis Probabilities", num_top_classes=2),
    title="Alzheimer's Disease MRI Classifier",
    description="""
    Upload a T1-weighted brain MRI scan in NIfTI format and enter clinical information to classify cognitive status.
    
    **Model:** MedicalNet ResNet10 + Clinical Feature Fusion, trained on ADNI dataset (330 subjects, binary CN vs Dementia)
    
    **Clinical features:** MMSE score, CDR sum of boxes, age, sex, education
    
    ⚠️ *For research purposes only. Not a clinical diagnostic tool.*
    """,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        font=gr.themes.GoogleFont("Inter")
    ),
    article="""
    <p style='text-align: center'>
    Built by <a href='https://github.com/ArinehKhachikian'>Arineh Khachikian</a> | 
    <a href='https://github.com/ArinehKhachikian/alzheimer-mri-progression'>GitHub</a> | 
    Data: ADNI (adni.loni.usc.edu)
    </p>
    """
)

if __name__ == "__main__":
    demo.launch()
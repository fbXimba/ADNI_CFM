import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
from pathlib import Path

class ADNIDataset(Dataset):
    """
    Class for loading the ADNI dataset. Expects a directory structure like:
    dataset_dir/
        image/
            subject1_brain.nii.gz
            subject2_brain.nii.gz
        mask/
            subject1_mask.nii.gz
            subject2_mask.nii.gz
        diagnosis/
            train_subjects.csv
            
    Note:
        For validation dataset same structure but with validation_subjects.csv in diagnosis/ folder.
    """

    def __init__(self, dataset_dir, split="train"):
        dataset_dir = Path(dataset_dir)

        self.image_dir = os.path.join(dataset_dir, "image")
        self.mask_dir = os.path.join(dataset_dir, "mask")
        df = pd.read_csv(os.path.join(dataset_dir, "diagnosis", f"{split}_subjects.csv"))
        self.ids = df["Subject"].tolist()
        self.labels = df["Diagnosis"].tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        image = nib.load(os.path.join(self.image_dir, f"{id}_brain.nii.gz")).get_fdata()
        mask = nib.load(os.path.join(self.mask_dir, f"{id}_mask.nii.gz")).get_fdata()

        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, D, H, W)
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # (1, D, H, W)

        diagnosis = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )  # scalar

        return {
            "Subject": id,
            "image": image,
            "mask": mask,
            "diagnosis": diagnosis
        }

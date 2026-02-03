import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
from pathlib import Path

class Dataset(Dataset):
    def __init__(self, dataset_dir):
        dataset_dir = Path(dataset_dir)

        self.image_dir = os.path.join(dataset_dir, "image")
        self.mask_dir = os.path.join(dataset_dir, "mask")
        df = pd.read_csv(os.path.join(dataset_dir, "diagnosis", "train_subjects.csv"))
        self.ids = df["case_id"].tolist()
        self.labels = dict(zip(df["case_id"], df["diagnosis"]))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cid = self.ids[idx]

        image = nib.load(os.path.join(self.image_dir, f"{cid}.nii.gz")).get_fdata()
        mask = nib.load(os.path.join(self.mask_dir, f"{cid}.nii.gz")).get_fdata()

        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, D, H, W)
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # (1, D, H, W)

        diagnosis = torch.tensor(
            self.labels[cid],
            dtype=torch.long
        )  # scalar

        return {
            "image": image,
            "mask": mask,
            "diagnosis": diagnosis
        }

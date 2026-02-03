"""
CFM Training Script for ADNI Dataset
"""

# TODO: create proper enviroment temp: env-tesi

# Libraries
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # TODO: set specific GPU if multiple available
os.environ["CUDA_VISIBLE_DEVICES"]="1" # TODO: set specific GPU if multiple available

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher as CFM
from model.unet_ADNI import create_model 
from model.trainer import Trainer   # import Trainer class from trainer.py  
from dataset import Dataset
from torch.utils.data import DataLoader
import datetime
import yaml
import wandb


# Hyperparameters and settings: implemented as argparse arguments later
epochs = 10
dataset_dir = "../ADNI_split/ADNI_training_dataset/"
validation_dataset_dir = "../ADNI_split/ADNI_validation_dataset/"
batch_size = 4
num_workers = 4
input_size = 128
num_channels = 64
num_res_blocks = 2
in_channels = 2  # image + mask
out_channels = 1  # velocity field
num_classes = 3  # CN, MCI, AD
save_every = 100
lr = 2e-4
loss = "le"
warmup_steps = 0
lr_scheduler = "cos"
lr_min = 2e-7
#gammadecay = 0.9999
#pl_factor = 0.5
#pl_patience = 500
results_dir = "./results_CFM_ADNI"
key_dir = "./key.yaml"  # wandb key file

# Create results directory with timestamp
now=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir= os.path.join(results_dir, now)
os.makedirs(run_dir, exist_ok=True)

# Initialize wandb
if os.path.exists(key_dir):
    # Load API key
    with open("key.yaml") as file:
        config=yaml.safe_load(file)
    key=config["wandb"]["key"]
    # Login
    wandb.login(key=key)
    wb=now
else:
    print("Wandb key file not found. Proceeding without wandb logging.")
    wb=False


# Dataset and DataLoader
dataset = Dataset(dataset_dir) # from dataset.py

loader = DataLoader( # from torch.utils.data
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True, # to speed up transfer to GPU?
    persistent_workers=True # keep workers alive between epochs?
)

# Validation set dataset and DataLoader
if validation_dataset_dir is not None:
    val_dataset = Dataset(validation_dataset_dir) # validation dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

#check 
batch = next(iter(loader))

for k, v in batch.items(): # print batch keys and tensor shapes
    print(k, v.shape, v.dtype)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model

in_channels = 2 # mask + noise image when conditioned
out_channels = 1 # only one channel = one output velocity field

model = create_model(
   input_size, 
   num_channels, 
   num_res_blocks, 
   class_cond=True, # condition on diagnosis
   in_channels=in_channels, 
   out_channels=out_channels,
   num_classes=num_classes # specify number of classes
   ).to(device)

# Trainer

trainer = Trainer(
    model=model,
    loader=loader,
    val_loader=(val_loader if 'val_loader' in locals() else None),
    device=device,
    lr=lr,
    epochs=epochs,
    save_every=save_every,
    results_dir=run_dir,
    loss_type=loss,
    warmup_steps=warmup_steps,
    scheduler_type=lr_scheduler,
    lr_min=lr_min,
    wb=wb
    )

# Start training
trainer.train()

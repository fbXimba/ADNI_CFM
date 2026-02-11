"""
CFM Training Script for ADNI Dataset
"""

# TODO: create proper enviroment temp: env-tesi, epochs nomenclature used improperly? check training loop
# NOTE: if lr shedule = ReduceLROnPlateau need validation set for validation loss for scheduler step

# Libraries
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # TODO: set specific GPU if multiple available
os.environ["CUDA_VISIBLE_DEVICES"]="1" # TODO: set specific GPU if multiple available

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
#from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher as CFM
from model.unet_ADNI import create_model 
from model.trainer import Trainer   # import Trainer class from trainer.py  
from dataset import Dataset
from torch.utils.data import DataLoader
import datetime
import yaml
import wandb

# Load configuration file: directoories, parameters and wandb
with open("config.yaml") as f:
    config = yaml.safe_load(f)
dir=config["directories"]
params=config["parameters"]

# Hyperparameters and settings
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=params["epochs"], help="Number of training epochs")
parser.add_argument("--dataset_dir", type=str, default=dir["dataset_dir"], help="Path to the training dataset")
parser.add_argument("--validation_dataset_dir", type=str, default=dir["validation_dataset_dir"], help="Path to the validation dataset")
parser.add_argument("--batch_size", type=int, default=params["batch_size"], help="Batch size for training")
parser.add_argument("--num_workers", type=int, default=params["num_workers"], help="Number of workers for data loading")
parser.add_argument("--input_size", type=int, default=params["input_size"], help="Input size for the model")
parser.add_argument("--num_channels", type=int, default=params["num_channels"], help="Number of channels in the model")
parser.add_argument("--num_res_blocks", type=int, default=params["num_res_blocks"], help="Number of residual blocks in the model")
parser.add_argument("--in_channels", type=int, default=params["in_channels"], help="Number of input channels (image + mask)")
parser.add_argument("--out_channels", type=int, default=params["out_channels"], help="Number of output channels (velocity field)")
parser.add_argument("--num_classes", type=int, default=params["num_classes"], help="Number of classes (CN, MCI, AD)")
parser.add_argument("--save_every", type=int, default=params["save_every"], help="Save model every n steps")
parser.add_argument("--lr", type=float, default=params["lr"], help="Learning rate")
parser.add_argument("--loss_type", type=str, default=params["loss_type"], help="Loss function type: \"l1\", \"l2\", \"le\" implemented")
parser.add_argument("--warmup_steps", type=int, default=params["warmup_steps"], help="Number of warmup steps")
parser.add_argument("--lr_scheduler", type=str, default=params["lr_scheduler"], help="Learning rate scheduler: \"cos\", \"exp\", \"plateau\" implemented")
parser.add_argument("--lr_min", type=float, default=params["lr_min"], help="Minimum learning rate")
parser.add_argument("--gamma_decay", type=float, default=params["gamma_decay"], help="Gamma decay for learning rate with exponential decay scheduler")
parser.add_argument("--pl_factor", type=float, default=params["pl_factor"], help="Factor applied for learning rate with plateau scheduler")
parser.add_argument("--pl_patience", type=int, default=params["pl_patience"], help="Patience for learning rate with plateau scheduler")
parser.add_argument("--use_ema", type=bool, default=params["use_ema"], help="Use EMA")
parser.add_argument("--ema_decay", type=float, default=params["ema_decay"], help="EMA decay")
parser.add_argument("--update_ema_every", type=int, default=params["update_ema_every"], help="Update EMA every n steps")
parser.add_argument("--grad_norm", type=float, default=params["grad_norm"], help="Gradient clipping norm max, set to None to disable")
parser.add_argument("--results_dir", type=str, default=dir["results_dir"], help="Directory to save runs' results")
parser.add_argument("--key_dir", type=str, default=dir["key_dir"], help="Weight and Biases key")

args = parser.parse_args()

# Create results directory with timestamp
now=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir= os.path.join(args.results_dir, now)
os.makedirs(run_dir, exist_ok=True)

# Initialize wandb
if args.key_dir is not None:
    # Set environment variables to disable unwanted wandb features
    try:
        os.environ["WANDB_DISABLE_CODE"] = "true"  # no code snapshot
    except Exception as e:
        print(f"Error setting wandb environment variables: {e}")
    try:        
        os.environ["WANDB_WATCH"] = "false"                # no model graph logging
    except Exception as e:
        print(f"Error setting wandb environment variables: {e}")
    # Load API key
    key=config["wandb"]["key"]
    # Login
    wandb.login(key=key)
else:
    print("Wandb key file not found: proceeding without wandb logging.")


# Dataset and DataLoader
dataset = Dataset(args.dataset_dir) # from dataset.py

loader = DataLoader( # from torch.utils.data
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True, # to speed up transfer to GPU
    #persistent_workers=True, # keep workers alive between epochs?
    drop_last = True
)

# Validation dataset and DataLoader
if os.path.exists(args.validation_dataset_dir):
    val_dataset = Dataset(args.validation_dataset_dir) # validation dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        #persistent_workers=True,
        drop_last = True
    )

#check 
batch = next(iter(loader))

#for k, v in batch.items(): # print batch keys and tensor shapes
#    print(k, v.shape, v.dtype)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model

in_channels = 2 # mask + noise image when conditioned
out_channels = 1 # only one channel = one output velocity field

model = create_model(
   args.input_size, 
   args.num_channels, 
   args.num_res_blocks, 
   class_cond=True, # condition on diagnosis
   in_channels=in_channels, 
   out_channels=out_channels,
   num_classes=args.num_classes # specify number of classes
   ).to(device)

# Trainer

trainer = Trainer(
    model=model,
    loader=loader,
    val_loader=(val_loader if 'val_loader' in locals() else None),
    device=device,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    save_every=args.save_every,
    results_dir=run_dir,
    loss_type=args.loss_type,
    use_ema=args.use_ema,
    ema_decay=args.ema_decay,
    update_ema_every=args.update_ema_every,
    warmup_steps=args.warmup_steps,
    scheduler_type=args.lr_scheduler,
    lr_min=args.lr_min,
    gammadecay=args.gammadecay, # for exponential decay scheduler
    pl_factor=args.pl_factor, # for ReduceLROnPlateau scheduler
    pl_patience=args.pl_patience, # for ReduceLROnPlateau scheduler
    wb_run=(now if args.key_dir is not None else None), # use timestamp as wandb run name if wandb logging enabled
    grad_norm=args.grad_norm #gradien clipping norm max
    )

# Start training
trainer.train()

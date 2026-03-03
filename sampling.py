import os
import numpy as np
import torch
import torchdiffeq
import nibabel as nib
from model.unet_ADNI import create_model
import argparse
import yaml
import datetime

with open("config.yaml") as f:
    config = yaml.safe_load(f)
dirs = config["directories"]
samp = config["sampling"]
params = config["parameters"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default=dirs["test_dataset_dir"], help="Path to the directory containing mask files")
parser.add_argument("--num_samples", type=int, default=samp["num_samples"], help="Number of samples to generate")
parser.add_argument("--label", type=str, default=samp["label"], help="Diagnosis label: (\"CN\", \"MCI\", \"AD\")")
parser.add_argument("--seed", type=int, default=samp["seed"], help="Random seed for sampling")
parser.add_argument("--sample_dir", type=str, default=dirs["gen_samples_dir"], help="Directory to save sampled data")
parser.add_argument("--checkpoint", type=int, default=samp["checkpoint"], help="Checkpoint step to load the model from")
parser.add_argument("--checkpoints_dir", type=str, default=dirs["checkpoints_dir"], help="Directory of checkpoints")
parser.add_argument("--input_size", type=int, default=params["input_size"], help="Input size for the model")
parser.add_argument("--num_channels", type=int, default=params["num_channels"], help="Number of channels in the model")
parser.add_argument("--num_res_blocks", type=int, default=params["num_res_blocks"], help="Number of residual blocks in the model")
parser.add_argument("--in_channels", type=int, default=params["in_channels"], help="Number of input channels (image + mask)")
parser.add_argument("--out_channels", type=int, default=params["out_channels"], help="Number of output channels (velocity field)")
parser.add_argument("--num_classes", type=int, default=params["num_classes"], help="Number of classes (CN, MCI, AD)")
parser.add_argument("--ema", type=bool, default=samp["ema"], help="Whether to use EMA weights for sampling")
parser.add_argument("--mask_id", type=str, default=samp["mask_id"], help="Subject ID to use for mask conditioning, if None uses random masks from directory")

args = parser.parse_args()

print("NOTE: assuming parameters correspond to the trained model chosen!!")

###############################################################################

def load_mask_with_affine(mask_path):
    """
    Load a mask file with its affine transformation
    Args:
    -----
        mask_path: str
            path to the mask file (.nii.gz)
    Returns:
    --------
        mask: torch.Tensor
            mask tensor (1, D, H, W)
        affine: np.ndarray
            affine transformation matrix (4, 4)
    """
    nifti_img = nib.load(mask_path)
    mask_data = nifti_img.get_fdata()
    affine = nifti_img.affine
    
    mask = torch.from_numpy(mask_data).float().unsqueeze(0)  # (1, D, H, W)
    return mask, affine

def get_available_masks(mask_dir):
    """
    Get all available mask files from directory
    Args:
    -----
        mask_dir: str
            directory containing mask files
    Returns:
    --------
        mask_files: list
            list of tuples (subject_id, mask_path)
    """
    mask_files = []
    for f in sorted(os.listdir(mask_dir)):
        if f.endswith('_mask.nii.gz'):
            subject_id = f.replace('_mask.nii.gz', '')
            mask_files.append((subject_id, os.path.join(mask_dir, f)))
    return mask_files

def load_trained_model(checkpoint_dir, checkpoint_step, input_size, num_channels, num_res_blocks, in_channels, out_channels, num_classes, ema, device):
    """
    Create model and load weights from checkpoint
    Args:
    -----
        checkpoint_dir: str
            directory where checkpoints are saved
        checkpoint_step: int
            step number of the checkpoint to load
        input_size: int
            input size for the model
        num_channels: int
            number of channels in the model     
        num_res_blocks: int
            number of residual blocks in the model
        in_channels: int
            number of input channels (image + mask)
        out_channels: int
            number of output channels (velocity field)
        num_classes: int
            number of classes (CN, MCI, AD)
        ema: bool
            whether to use EMA weights for loading the model
        device: torch.device
            device to load the model on
    Returns:
    --------
        model: torch.nn.Module
            model with loaded weights from checkpoint
    """

    # Model
    model = create_model(
       input_size, 
       num_channels, 
       num_res_blocks, 
       #channel_mult=?, # default is 128, (1,1,2,3,4) in create_model in unet_ADNI.py
       class_cond=True, # condition on diagnosis
       in_channels=in_channels, 
       out_channels=out_channels,
       num_classes=num_classes # specify number of classes
       ).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_step}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model weights: use EMA if available and requested
    if ema and 'ema_model' in checkpoint:
        model.load_state_dict(checkpoint['ema_model'])
        print(f"Loaded EMA model from: {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded main model from: {checkpoint_path}")
    
    model.eval()  # Set to evaluation mode
    
    # Print validation metrics if available
    if checkpoint.get('ema_val_loss') is not None:
        print(f"  EMA val loss: {checkpoint['ema_val_loss']:.4f}")
    if checkpoint.get('val_loss') is not None:
        print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    return model

def generate_sample(model, noise, mask, diagnosis, device):
    """
    Generates a sample from the model given noise, mask, and diagnosis conditioning
    Args:   
    -----
        model: torch.nn.Module
            trained model to sample from
        noise: torch.Tensor
            random noise input (1, 1, D, H, W)
        mask: torch.Tensor
            spatial constraint mask (1, 1, D, H, W)
        diagnosis: torch.Tensor
            target diagnosis label for conditioning (1,)
        device: torch.device
            device to perform computation on
    Returns:
    --------       
        generated_sample: torch.Tensor
            generated sample at t=1 (1, 1, D, H, W)
    """

    with torch.no_grad():
        # Solve ODE to generate sample: noise evolves and mask stays constant
        traj = torchdiffeq.odeint( 
            lambda t, x: model(torch.cat([x, mask], dim=1), t.reshape(-1), diagnosis), # concatenate evolving x with fixed mask
            noise, # initial condition (t=0)
            torch.linspace(0, 1, 2, device=device), # time points to solve for: from t=0 (noise) to t=1 (generated sample)
            atol=1e-4, # absolute tolerance: smaller values = more accurate but slower
            rtol=1e-4, # relative tolerance: smaller values = more accurate but slower
            method="dopri5" # Runge-Kutta method : order 5, adaptive step size
        )
        
    return traj[-1]  # Return the last time step of the trajectory traj[-1] = generated sample at t=1

def sample_from_mask(model, mask_path, num_samples, sample_dir, target_label, seed, device):
    """
    Samples using fixed mask and target diagnosis
    Args:
    -----
        model: torch.nn.Module
            trained model to sample from
        mask_path: str
            path to mask file (.nii.gz)
        num_samples: int
            number of samples to generate
        sample_dir: str
            directory to save the generated samples
        target_label: int
            target diagnosis label for conditional generation
        seed: int
            random seed for reproducibility
        device: torch.device
            device to perform computation on  
    Returns:
    --------
        None (saves generated samples to specified directory)
    """

    # Load mask with its affine transformation matrix
    mask, affine = load_mask_with_affine(mask_path)
    subject_id = os.path.basename(mask_path).replace('_mask.nii.gz', '')

    # Mask and diagnosis to device
    mask = mask.to(device).unsqueeze(0)  # (1, 1, D, H, W)
    diagnosis = torch.tensor([target_label], dtype=torch.long).to(device)  # (1,)
    
    for i in range(num_samples):
        # Start from noise with seed for reproducibility
        seed_i = seed + i  # different seed for each sample
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)

        # Generate noise input: *mask.shape[2:] = (D, H, W)
        noise = torch.randn(1, 1, *mask.shape[2:]).to(device)  # (1, 1, D, H, W)
        
        # Generate sample with ODE solver (no grad)
        generated_sample = generate_sample(model, noise, mask, diagnosis, device)

        # Save generated sample with correct affine matrix
        save_path = os.path.join(sample_dir, f"{subject_id}_sampled_{target_label}_{seed_i}.nii.gz")
        nib.save(nib.Nifti1Image(generated_sample.squeeze().cpu().numpy(), affine=affine), save_path)
        print(f"Saved: {save_path}")

    return

def sample_model(model, mask_dir, num_samples, sample_dir, target_label, seed, device):
    """
    Samples without fixed mask but with target diagnosis conditioning
    Args:
    -----
        model: torch.nn.Module
            trained model to sample from
        mask_dir: str
            directory containing mask files
        num_samples: int
            number of samples to generate
        sample_dir: str
            directory to save the generated samples
        target_label: int
            target diagnosis label for conditional generation
        seed: int
            random seed for reproducibility 
        device: torch.device
            device to perform computation on
    Returns:
    --------
        None (saves generated samples to specified directory)
    """

    # Get available masks from directory
    available_masks = get_available_masks(mask_dir)
    if not available_masks:
        raise ValueError(f"No mask files found in {mask_dir}")

    # Diagnosis conditioning to device
    diagnosis = torch.tensor([target_label], dtype=torch.long).to(device)  # (1,)

    for i in range(num_samples):
        # Sample random mask from available masks (truly random, not seeded)
        mask_idx = np.random.randint(len(available_masks))
        subject_id, mask_path = available_masks[mask_idx]
        
        # Load mask with its affine transformation matrix
        mask, affine = load_mask_with_affine(mask_path)
        
        # Mask to device
        mask = mask.to(device).unsqueeze(0)  # (1, 1, D, H, W)
        
        # Set seed for noise generation (reproducibility for each sample index)
        seed_i = seed + i  # different seed for each sample
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)
        
        # Generate noise input with seed: *mask.shape[2:] = (D, H, W)
        noise = torch.randn(1, 1, *mask.shape[2:]).to(device)  # (1, 1, D, H, W)
        
        # Generate sample with ODE solver (no grad)
        generated_sample = generate_sample(model, noise, mask, diagnosis, device)

        # Save generated sample with correct affine matrix
        save_path = os.path.join(sample_dir, f"{subject_id}_sampled_{target_label}_{seed_i}.nii.gz")
        nib.save(nib.Nifti1Image(generated_sample.squeeze().cpu().numpy(), affine=affine), save_path)
        print(f"Saved: {save_path}")

    return

###############################################################################
# Main sampling code

# Create sample directory with timestamp
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sample_dir = os.path.join(args.sample_dir, now)
os.makedirs(sample_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target diagnosis label for generation
label_map = {"CN": 0, "MCI": 1, "AD": 2}
target_label = label_map[args.label] # integer label for conditional generation

# Load trained model
model = load_trained_model(args.checkpoints_dir, args.checkpoint, args.input_size, args.num_channels, args.num_res_blocks, args.in_channels, args.out_channels, args.num_classes, args.ema, device)

if args.mask_id is not None:
    print(f"Sampling with fixed mask from subject ID: {args.mask_id}")
    mask_path = os.path.join(args.dataset_dir, "mask", f"{args.mask_id}_mask.nii.gz")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    sample_from_mask(model, mask_path, args.num_samples, sample_dir, target_label, args.seed, device)
else:
    print("Sampling without fixed mask (random masks from directory)")
    mask_dir = os.path.join(args.dataset_dir, "mask")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    sample_model(model, mask_dir, args.num_samples, sample_dir, target_label, args.seed, device)

import os
import numpy as np
import torch
import torchdiffeq
import nibabel as nib
from model.unet_ADNI import create_model
from typing import List, Tuple
import argparse
import yaml
import datetime

# dictionary with label mapping for filename generation
idx_to_label = {
    0: "CN",
    1: "MCI",
    2: "AD"
}

label_to_idx = {
    "CN": 0,
    "MCI": 1,
    "AD": 2
}

################################################################################
# FUNCTIONS 
###############################################################################

def load_mask_with_affine(mask_path: str) -> Tuple[torch.Tensor, np.ndarray]:
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

def get_available_masks(mask_dir: str) -> List[Tuple[str, str]]:
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

def load_trained_model(checkpoint_dir: str, checkpoint_step: int, input_size: int, num_channels: int, num_res_blocks: int, in_channels: int, out_channels: int, num_classes: int, ema: bool, device: torch.device) -> torch.nn.Module:
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

def generate_sample(model: torch.nn.Module, noise: torch.Tensor, mask: torch.Tensor, diagnosis: torch.Tensor, device: torch.device) -> torch.Tensor:
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
        # Keep mask and diagnosis packed together in a single conditioning tensor for OT alignement
        B = mask.shape[0]
        diagnosis_scalar = diagnosis.view(B, -1)[:, :1]
        diagnosis_map = diagnosis_scalar.to(mask.dtype).view(B, 1, 1, 1, 1).expand(-1, 1, *mask.shape[2:])
        cond = torch.cat([mask, diagnosis_map], dim=1)

        # Recover conditioning components from packed tensor
        mask_cond = cond[:, :1]
        diagnosis_cond = cond[:, 1, 0, 0, 0].to(diagnosis.dtype)

        # Solve ODE to generate sample: noise evolves and mask stays constant
        traj = torchdiffeq.odeint( 
            lambda t, x: model(torch.cat([x, mask_cond], dim=1), t.reshape(-1), diagnosis_cond), # concatenate evolving x with fixed mask
            noise, # initial condition (t=0)
            torch.linspace(0, 1, 2, device=device), # time points to solve for: from t=0 (noise) to t=1 (generated sample)
            atol=1e-4, # absolute tolerance: smaller values = more accurate but slower
            rtol=1e-4, # relative tolerance: smaller values = more accurate but slower
            method="dopri5" # Runge-Kutta method : order 5, adaptive step size
        )
        
    return traj[-1]  # Return the last time step of the trajectory traj[-1] = generated sample at t=1

def generate_noise_with_seed(shape: Tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    """
    Generate noise tensor with fixed seed for reproducibility    
    Args:
    -----
        shape: tuple
            shape of noise tensor (e.g., (1, 1, D, H, W))
        seed: int
            random seed for noise generation
        device: torch.device
            device to create tensor on
    Returns:
    --------
        noise: torch.Tensor
            random noise tensor
    """
    torch.manual_seed(seed)
    return torch.randn(shape, device=device)

def prepare_mask_for_sampling(mask_path: str, device: torch.device) -> Tuple[torch.Tensor, np.ndarray, str]:
    """
    Load mask and prepare for sampling    
    Args:
    -----
        mask_path: str
            path to mask file (.nii.gz)
        device: torch.device
            device to move mask to
    Returns:
    --------
        mask: torch.Tensor
            mask tensor (1, 1, D, H, W)
        affine: np.ndarray
            affine transformation matrix
        subject_id: str
            subject identifier extracted from filename
    """
    mask, affine = load_mask_with_affine(mask_path)
    subject_id = os.path.basename(mask_path).replace('_mask.nii.gz', '')
    mask = mask.to(device).unsqueeze(0)  # (1, 1, D, H, W)
    return mask, affine, subject_id

def prepare_diagnosis_conditioning(target_label: int, device: torch.device) -> torch.Tensor:
    """
    Create diagnosis tensor for conditioning    
    Args:
    -----
        target_label: int
            target diagnosis label index
        device: torch.device
            device to create tensor on
    Returns:
    --------
        diagnosis: torch.Tensor
            diagnosis tensor (1,)
    """
    return torch.tensor([target_label], dtype=torch.long).to(device)

def save_samples(samples_data: List[Tuple[torch.Tensor, str, int, int, np.ndarray]], sample_dir: str) -> List[str]:
    """
    Saves generated samples    
    Args:
    -----
        samples_data: list of tuples
            list of (generated_sample, subject_id, target_label, seed_i, affine) tuples
        sample_dir: str
            directory to save the generated samples
    Returns:
    --------
        saved_paths: list
            list of paths where samples were saved
    """
    saved_paths = []
    for generated_sample, subject_id, target_label, seed_i, affine in samples_data:
        # Save generated sample with correct affine matrix
        save_path = os.path.join(sample_dir, f"{subject_id}_sampled_{idx_to_label[target_label]}_{seed_i}.nii.gz")
        nib.save(nib.Nifti1Image(generated_sample.squeeze().cpu().numpy(), affine=affine), save_path)
        print(f"Saved: {save_path}")
        saved_paths.append(save_path)
    
    return saved_paths

def sample_from_mask(model: torch.nn.Module, mask_path: str, num_samples: int, target_label: int, seed: int, device: torch.device) -> List[Tuple[torch.Tensor, str, int, int, np.ndarray]]:
    """
    Generates samples using fixed mask and target diagnosis
    Args:
    -----
        model: torch.nn.Module
            trained model to sample from
        mask_path: str
            path to mask file (.nii.gz)
        num_samples: int
            number of samples to generate
        target_label: int
            target diagnosis label for conditional generation
        seed: int
            random seed for reproducibility
        device: torch.device
            device to perform computation on  
    Returns:
    --------
        samples_data: list of tuples
            list of (generated_sample, subject_id, target_label, seed_i, affine) tuples
    """
    mask, affine, subject_id = prepare_mask_for_sampling(mask_path, device)
    diagnosis = prepare_diagnosis_conditioning(target_label, device)
    
    samples_data = []
    for i in range(num_samples):
        seed_i = seed + i * 6  # different seed for each sample: increment of 6
        noise = generate_noise_with_seed((1, 1, *mask.shape[2:]), seed_i, device)
        generated_sample = generate_sample(model, noise, mask, diagnosis, device)
        samples_data.append((generated_sample, subject_id, target_label, seed_i, affine))

    return samples_data

def sample_model(model: torch.nn.Module, mask_dir: str, num_samples: int, target_label: int, seed: int, device: torch.device) -> List[Tuple[torch.Tensor, str, int, int, np.ndarray]]:
    """
    Generates samples with random masks and target diagnosis conditioning    
    Args:
    -----
        model: torch.nn.Module
            trained model to sample from
        mask_dir: str
            directory containing mask files
        num_samples: int
            number of samples to generate
        target_label: int
            target diagnosis label for conditional generation
        seed: int
            random seed for reproducibility 
        device: torch.device
            device to perform computation on
    Returns:
    --------
        samples_data: list of tuples
            list of (generated_sample, subject_id, target_label, seed_i, affine) tuples
    """
    available_masks = get_available_masks(mask_dir)
    if not available_masks:
        raise ValueError(f"No mask files found in {mask_dir}")

    diagnosis = prepare_diagnosis_conditioning(target_label, device)
    samples_data = []
    
    for i in range(num_samples):
        # Sample random mask from available masks
        mask_idx = np.random.randint(len(available_masks))
        subject_id, mask_path = available_masks[mask_idx]
        
        mask, affine, _ = prepare_mask_for_sampling(mask_path, device)
        seed_i = seed + i * 6
        noise = generate_noise_with_seed((1, 1, *mask.shape[2:]), seed_i, device)
        generated_sample = generate_sample(model, noise, mask, diagnosis, device)
        samples_data.append((generated_sample, subject_id, target_label, seed_i, affine))

    return samples_data

###############################################################################
# MAIN SAMPLING CODE
###############################################################################

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # TODO: set specific GPU if multiple available
    os.environ["CUDA_VISIBLE_DEVICES"]="1" # TODO: set specific GPU if multiple available
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True" # to allow memory fragmentation and reduce OOM errors
    
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
    parser.add_argument("--run", type=str, default=samp["run"], help="Run identifier for checkpoint loading")
    parser.add_argument("--checkpoints_dir", type=str, default=dirs["checkpoints_dir"], help="Directory of checkpoints")
    parser.add_argument("--input_size", type=int, default=params["input_size"], help="Input size for the model")
    parser.add_argument("--num_channels", type=int, default=params["num_channels"], help="Number of channels in the model")
    parser.add_argument("--num_res_blocks", type=int, default=params["num_res_blocks"], help="Number of residual blocks in the model")
    parser.add_argument("--in_channels", type=int, default=params["in_channels"], help="Number of input channels (image + mask)")
    parser.add_argument("--out_channels", type=int, default=params["out_channels"], help="Number of output channels (velocity field)")
    parser.add_argument("--num_classes", type=int, default=params["num_classes"], help="Number of classes (CN, MCI, AD)")
    parser.add_argument("--ema", type=bool, default=samp["ema"], help="Whether to use EMA weights for sampling")
    parser.add_argument("--mask_id", type=str, default=samp["mask_id"], help="Subject ID to use for mask conditioning, if None uses random masks from directory")
    #parser.add_argument("--csv_file_dataset", type=str, default=samp["csv_file_dataset"], help="CSV file coupling seeds, subjects and diagnosis for dataset creation")
    
    args = parser.parse_args()
      
    print("NOTE: assuming parameters correspond to the trained model chosen!!")
    
    # Create sample directory with timestamp
    #now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = os.path.join(args.sample_dir, args.run, args.checkpoint)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Target diagnosis label for generation
    target_label = label_to_idx[args.label] # integer label for conditional generation
    
    # Load trained model
    checkpoint_path = os.path.join(args.checkpoints_dir, args.run)
    model = load_trained_model(checkpoint_path, args.checkpoint, args.input_size, args.num_channels, args.num_res_blocks, args.in_channels, args.out_channels, args.num_classes, args.ema, device)
    
    # Generate samples
    if args.mask_id is not None:
        print(f"Sampling with fixed mask from subject ID: {args.mask_id}")
        mask_path = os.path.join(args.dataset_dir, "mask", f"{args.mask_id}_mask.nii.gz")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        samples_data = sample_from_mask(model, mask_path, args.num_samples, target_label, args.seed, device)
    else:
        print("Sampling without fixed mask (random masks from directory)")
        mask_dir = os.path.join(args.dataset_dir, "mask")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        samples_data = sample_model(model, mask_dir, args.num_samples, target_label, args.seed, device)
    
    # Save samples to output folder
    save_samples(samples_data, sample_dir)

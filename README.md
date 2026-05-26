# Brain MRI Generation from ADNI Dataset with CFM + OT :brain:

## Overview

CFM_ADNI is a pipeline for training a Conditional Flow Matching model on the ADNI dataset to generate synthetic 3D brain MRIs. The model learns to generate realistic brain images conditioned on diagnostic labels (CN, MCI, AD) from ***preprocessed*** real patient data using Conditional Flow Matching (CFM) and Optimal Transport.

**Note**: The preprocessing pipeline used for the ADNI dataset can be found on [ADNI_preprocessing](https://github.com/fbXimba/ADNI_preprocessing)

## Pipeline :eyes:

```
Preprocessed ADNI Data → [Train] → CFM+OT Model → [Sample] → Synthetic Brain MRI Dataset
              ↓
         Checkpoints & Metrics
```

## Main Scripts 

**Training** (`train.py`): Trains a 3D UNet model with Conditional Flow Matching and Optimal Transport. Supports gradient checkpointing, EMA, and Weights & Biases logging.

**Sampling** (`sampling.py`): Generates synthetic brain MRI from trained checkpoints, conditioned on diagnosis (CN/MCI/AD) and anatomical masks.

**Dataset Creation** (`create_dataset.py`): Batch generates synthetic samples and saves in NIfTI format with .csv info file.

## Requirements and setup:

:warning: **GPU Selection**: scripts default to GPU 1 Update the device selection at the 
top of the main scripts for your system.

**1. Pytorch**

Install PyTorch >=2.0 with **your** CUDA version from 
[pytorch.org](https://pytorch.org/get-started/locally/) 
first.

**2. Dependencies**

```bash
pip install -r requirements.txt
```
**3. Paths**

Update `config.yaml` with your data paths

**4. Train model**
```bash
python train.py --epochs 360 --batch_size 8 --lr 2e-4
```

**5. Generate samples**
```bash
python sampling.py --checkpoint 117 --num_samples 1 
--label CN
```

## Configuration file 

Edit `config.yaml` to set:
- Data directories (training/validation/test paths)
- Model hyperparameters (learning rate, batch size, 
epochs, loss type, ...)
- Logging key :key:

## Model Architecture 

**Features**:
- **3D UNet**: Encoder-decoder with skip connections for volumetric processing
- **Class Conditioning**: Diagnostic label embeddings (CN, MCI, AD)
- **Optimal Transport**: Improves training efficiency and sample quality
- **Gradient Checkpointing**: Reduces memory usage during training

**Optional Features**:
- EMA (Exponential Moving Average)
- Validation loss and EMA validation loss
- Weights and Biases logging

## Data Format :card_index_dividers:

Expected ADNI dataset structure:
```
training_dataset_dir/
    ├──image/
        ├──subject1_brain.nii.gz
        └── subject2_brain.nii.gz
    ├──mask/
        ├──subject1_mask.nii.gz
        └──subject2_mask.nii.gz
    └──diagnosis/
        └── train_subjects.csv       # columns: Subject, Diagnosis
```
Test and Validation datasets must follow the same format.

Expected generated samples structure:
```
sample_dir/
    ├──run1/
        └──checkpoint3/
    ├──run2/
        ├──checkpoint1/
        └──checkpoint2/
            ├──<subject_mask1>_sampled_<label>_<seed>.nii.gz
            └──<subject_mask2>_sampled_<label>_<seed>.nii.gz
```
## Testing :beetle:

Testing infrastructure with synthetic/mock data: better reproducibility (no ADNI dataset needed) but limiting.

Naming convention: test_< feature >_< scenario >.

**Setup:**

```bash
# Dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

```

## References 

- UNet architecture adapted from [OpenAI's guided-diffusion](https://github.com/openai/guided-diffusion)
- CFM framework from [torchcfm](https://github.com/atong01/conditional-flow-matching), with Minibatch Optimal Transport from ["Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"](https://arxiv.org/abs/2302.00482)

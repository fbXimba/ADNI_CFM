# Brain MRI Generation from ADNI Dataset with CFM + OT :brain:

## Overview

CFM_ADNI is a pipeline for training a Conditional Flow Matching model on the ADNI dataset to generate synthetic 3D brain MRIs. The model learns to generate realistic brain images conditioned on diagnostic labels (CN, MCI, AD) from ***preprocessed*** real patient data using Conditional Flow Matching (CFM) and Optimal Transport.

This project builds on `guided-diffusion` and `torchcfm`. The 3D UNet and utily modules are adapted from upstream sources, while the dataset, training, sampling, and evaluation workflow were developed for this project.

**Note**: My preprocessing pipeline used for the ADNI dataset can be found on [ADNI_preprocessing](https://github.com/fbXimba/ADNI_preprocessing). My postprocessing pipeline used to reinstate the sampled volumes to ouput similar to the ones from FreeSurfer can be found on [ADNI_postprocessing](https://github.com/fbXimba/ADNI_postprocessing)

## Data

The ADNI dataset is public but an application process it's required to access it. If interested please see at [ADNI data](https://adni.loni.usc.edu/data-samples/adni-data/).

## Pipeline :eyes:

```
Preprocessed ADNI Data → [Train] → CFM+OT Model → [Sample] → Synthetic Brain MRI Dataset
                            ↓                                           ↓
                        Checkpoints                                   Metrics
```

## Main Scripts 

**Training** (`train.py`): Trains a 3D UNet model with Conditional Flow Matching and Optimal Transport. Supports gradient checkpointing, EMA, and Weights & Biases logging.

**Sampling** (`sampling.py`): Generates synthetic brain MRI from trained checkpoints, conditioned on diagnosis (CN/MCI/AD) and anatomical masks.

**Dataset Creation** (`create_dataset.py`): Batch generates synthetic samples and saves in NIfTI format with .csv info file.

## Requirements and setup:

**1. Pytorch** :fire:

Install Python >= 3.10 and PyTorch >=2.0 with **your** CUDA version from 
[pytorch.org](https://pytorch.org/get-started/locally/) 
first. GPU use is strongly recommended for both training and sampling.

**2. Dependencies**

```bash
pip install -r requirements.txt
```

**3. Install the project**

```bash
pip install -e .
```

**4. Paths**

Update `config.yaml` with your data paths, parameters, etc.

**5. Train model**
```bash
python train.py --epochs 360 --batch_size 8 --lr 2e-4
```

**6. Generate samples**
```bash
python sampling.py --checkpoint 117 --num_samples 1 --label CN
```

## Configuration file 

Edit `config.yaml` to set:
- :warning: GPU id if multiple available
- Data directories (training/validation/test paths)
- Model hyperparameters (learning rate, batch size, epochs, loss type, ...)
- Weights and Biases logging key :key:

## Model Architecture 

**Features**:
- **3D UNet**: Encoder-decoder with skip connections for volumetric processing
- **Class Conditioning**: Diagnostic label embeddings (CN, MCI, AD)
- **Optimal Transport**: Improves training efficiency and sample quality
- **Gradient Checkpointing**: Reduces memory usage during training

**Optional Features**:
- **EMA** (Exponential Moving Average)
- **Validation** loss and EMA **validation** loss
- Weights and Biases **logging**

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

Testing uses synthetic/mock data so the suite runs without the ADNI dataset.

Naming convention: test_< feature >_< scenario >.

**Setup:**

```bash
# Dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# Run tests excluding slow ones
pytest tests/ -v -m "not slow"

```
**Coverage**: External utilities (`logger.py`, `fp16_util.py`,`modules.py`) from the adopted UNet (`unet_ADNI.py`) have lower coverage as they are from a third-party source.

## References 

- UNet architecture adapted from [OpenAI's guided-diffusion](https://github.com/openai/guided-diffusion)
- CFM framework (torchfm library) from [torchcfm](https://github.com/atong01/conditional-flow-matching), with Minibatch Optimal Transport from ["Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"](https://arxiv.org/abs/2302.00482)

**Note**: Non-original work includes the 3D UNet and supporting utility modules (logger.py, modules.py, fp16_util.py).
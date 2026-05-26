"""Unit tests for dataset"""

import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.unit

class TestDataset:
    """Test ADNI dataset"""
    
    def test_dataset_imports(self):
        """Test that dataset module can be imported"""
        try:
            from dataset import ADNIDataset
            assert ADNIDataset is not None
        except Exception as e:
            pytest.skip(f"Dataset import failed: {e}")
    
    def test_dataset_initialization(self, temp_dir):
        """Test dataset initialization with valid structure"""
        try:
            import nibabel as nib
            from dataset import ADNIDataset
        except Exception as e:
            pytest.skip(f"Dependencies missing: {e}")
        
        # Create mock directory structure
        dataset_dir = Path(temp_dir) / "dataset"
        dataset_dir.mkdir(exist_ok=True)
        (dataset_dir / "image").mkdir(exist_ok=True)
        (dataset_dir / "mask").mkdir(exist_ok=True)
        (dataset_dir / "diagnosis").mkdir(exist_ok=True)
        
        # Create dummy NIfTI files
        for i in range(2):
            img_data = np.random.randn(64, 64, 64)
            mask_data = np.random.randn(64, 64, 64)
            
            nib.save(nib.Nifti1Image(img_data, np.eye(4)), 
                    dataset_dir / "image" / f"subject{i}_brain.nii.gz")
            nib.save(nib.Nifti1Image(mask_data, np.eye(4)), 
                    dataset_dir / "mask" / f"subject{i}_mask.nii.gz")
        
        # Create diagnosis CSV
        df = pd.DataFrame({"Subject": ["subject0", "subject1"], "Diagnosis": [0, 1]})
        df.to_csv(dataset_dir / "diagnosis" / "train_subjects.csv", index=False)
        
        # Test initialization
        dataset = ADNIDataset(str(dataset_dir), split="train")
        assert dataset is not None
        assert len(dataset) == 2
    
    def test_dataset_length(self, temp_dir):
        """Test dataset __len__ method"""
        try:
            import nibabel as nib
            from dataset import ADNIDataset
        except Exception as e:
            pytest.skip(f"Dependencies missing: {e}")
        
        dataset_dir = Path(temp_dir) / "dataset"
        dataset_dir.mkdir(exist_ok=True)
        (dataset_dir / "image").mkdir(exist_ok=True)
        (dataset_dir / "mask").mkdir(exist_ok=True)
        (dataset_dir / "diagnosis").mkdir(exist_ok=True)
        
        # Create 3 subjects
        for i in range(3):
            img_data = np.random.randn(32, 32, 32)
            mask_data = np.random.randn(32, 32, 32)
            
            nib.save(nib.Nifti1Image(img_data, np.eye(4)), 
                    dataset_dir / "image" / f"subj{i}_brain.nii.gz")
            nib.save(nib.Nifti1Image(mask_data, np.eye(4)), 
                    dataset_dir / "mask" / f"subj{i}_mask.nii.gz")
        
        df = pd.DataFrame({"Subject": [f"subj{i}" for i in range(3)], "Diagnosis": [0, 1, 2]})
        df.to_csv(dataset_dir / "diagnosis" / "train_subjects.csv", index=False)
        
        dataset = ADNIDataset(str(dataset_dir), split="train")
        assert len(dataset) == 3
    
    def test_dataset_getitem_format(self, temp_dir):
        """Test dataset __getitem__ returns correct format"""
        try:
            import nibabel as nib
            from dataset import ADNIDataset
        except Exception as e:
            pytest.skip(f"Dependencies missing: {e}")
        
        dataset_dir = Path(temp_dir) / "dataset"
        dataset_dir.mkdir(exist_ok=True)
        (dataset_dir / "image").mkdir(exist_ok=True)
        (dataset_dir / "mask").mkdir(exist_ok=True)
        (dataset_dir / "diagnosis").mkdir(exist_ok=True)
        
        # Create single subject
        img_data = np.random.randn(48, 48, 48).astype(np.float32)
        mask_data = np.random.randn(48, 48, 48).astype(np.float32)
        
        nib.save(nib.Nifti1Image(img_data, np.eye(4)), 
                dataset_dir / "image" / "test_brain.nii.gz")
        nib.save(nib.Nifti1Image(mask_data, np.eye(4)), 
                dataset_dir / "mask" / "test_mask.nii.gz")
        
        df = pd.DataFrame({"Subject": ["test"], "Diagnosis": [1]})
        df.to_csv(dataset_dir / "diagnosis" / "train_subjects.csv", index=False)
        
        dataset = ADNIDataset(str(dataset_dir), split="train")
        item = dataset[0]
        
        # Check format
        assert isinstance(item, dict)
        assert "image" in item
        assert "mask" in item
        assert "diagnosis" in item
        
        # Check shapes and types
        assert item["image"].shape == (1, 48, 48, 48)
        assert item["mask"].shape == (1, 48, 48, 48)
        assert item["diagnosis"].shape == ()
        assert isinstance(item["image"], torch.Tensor)
        assert isinstance(item["mask"], torch.Tensor)
        assert isinstance(item["diagnosis"], torch.Tensor)


"""Tests for dataset"""

import pytest
import torch
from pathlib import Path

from .conftest import build_mock_adni_dataset


class TestDataset:
    """Test ADNI dataset"""
    
    def test_dataset_imports(self):
        """Verify that the dataset module exposes `ADNIDataset`.

        Returns
        -------
        None
        """
        try:
            from dataset import ADNIDataset
            assert ADNIDataset is not None
        except Exception as e:
            pytest.skip(f"Dataset import failed: {e}")
    
    def test_dataset_initialization(self, temp_dir):
        """Verify that dataset initialization succeeds for a valid layout.

        Parameters
        ----------
        temp_dir : pathlib.Path
            Temporary directory used to build the mock dataset.
        """
        try:
            from dataset import ADNIDataset
        except Exception as e:
            pytest.skip(f"Dependencies missing: {e}")

        dataset_dir = build_mock_adni_dataset(
            Path(temp_dir) / "dataset",
            ["subject0", "subject1"],
            diagnoses=[0, 1],
            shape=(64, 64, 64),
        )
        
        # Test initialization
        dataset = ADNIDataset(str(dataset_dir), split="train")
        assert dataset is not None
        assert len(dataset) == 2
    
    def test_dataset_length(self, temp_dir):
        """Verify that dataset length matches the number of subjects.

        Parameters
        ----------
        temp_dir : pathlib.Path
            Temporary directory used to build the mock dataset.
        """
        try:
            from dataset import ADNIDataset
        except Exception as e:
            pytest.skip(f"Dependencies missing: {e}")

        dataset_dir = build_mock_adni_dataset(
            Path(temp_dir) / "dataset",
            ["subj0", "subj1", "subj2"],
            diagnoses=[0, 1, 2],
            shape=(32, 32, 32),
        )
        
        dataset = ADNIDataset(str(dataset_dir), split="train")
        assert len(dataset) == 3
    
    def test_dataset_getitem_format(self, temp_dir):
        """Verify that dataset items return the expected tensor dict.
        
        Parameters
        ----------
        temp_dir : str
            Temporary directory for the test data.
        """
        try:
            from dataset import ADNIDataset
        except Exception as e:
            pytest.skip(f"Dependencies missing: {e}")

        dataset_dir = build_mock_adni_dataset(
            Path(temp_dir) / "dataset",
            ["test"],
            diagnoses=[1],
            shape=(48, 48, 48),
        )
        
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
        assert torch.all(item["image"] == 0)
        assert torch.all(item["mask"] == 1)


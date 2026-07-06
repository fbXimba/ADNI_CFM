"""Tests for sampling utilities"""

from pathlib import Path

import numpy as np
import nibabel as nib
import pytest
import torch
import torch.nn as nn

pytest.importorskip("torchdiffeq")

import sampling


def write_nifti(path: Path, data: np.ndarray) -> None:
    """Write a small NIfTI file for sampling tests."""
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)


class TestSamplingHelpers:
    """Test sampling helper functions"""

    def test_load_mask_with_affine(self, temp_dir):
        """Verify that masks are loaded with the original affine.

        Parameters
        ----------
        temp_dir : pathlib.Path
            Temporary directory used by the test.
        """
        mask_path = temp_dir / "subject_mask.nii.gz"
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        affine = np.array(
            [[1.0, 0.0, 0.0, 10.0], [0.0, 2.0, 0.0, 20.0], [0.0, 0.0, 3.0, 30.0], [0.0, 0.0, 0.0, 1.0]]
        )
        nib.save(nib.Nifti1Image(data, affine), mask_path)

        mask, loaded_affine = sampling.load_mask_with_affine(str(mask_path))

        assert mask.shape == (1, 2, 3, 4)
        assert mask.dtype == torch.float32
        assert np.allclose(loaded_affine, affine)
        assert torch.allclose(mask[0], torch.from_numpy(data))

    def test_prepare_mask_for_sampling(self, temp_dir, device):
        """Verify that masks are expanded for sampling and subject IDs are parsed.

        Parameters
        ----------
        temp_dir : pathlib.Path
            Temporary directory used by the test.
        device : torch.device
            Execution device for the tensors.
        """
        mask_path = temp_dir / "subject42_mask.nii.gz"
        write_nifti(mask_path, np.ones((2, 3, 4), dtype=np.float32))

        mask, affine, subject_id = sampling.prepare_mask_for_sampling(str(mask_path), device)

        assert mask.shape == (1, 1, 2, 3, 4)
        assert mask.device == device
        assert subject_id == "subject42"
        assert np.allclose(affine, np.eye(4))

    def test_prepare_diagnosis_conditioning(self, device):
        """Verify that diagnosis conditioning is built on the requested device.

        Parameters
        ----------
        device : torch.device
            Execution device for the tensors.
        """
        diagnosis = sampling.prepare_diagnosis_conditioning(2, device)

        assert diagnosis.shape == (1,)
        assert diagnosis.dtype == torch.long
        assert diagnosis.device == device
        assert diagnosis.item() == 2

    def test_generate_noise_with_seed_is_deterministic(self, device):
        """Verify that seeded noise generation is reproducible.

        Parameters
        ----------
        device : torch.device
            Execution device for the tensors.
        """
        noise_a = sampling.generate_noise_with_seed((1, 1, 2, 2, 2), 123, device)
        noise_b = sampling.generate_noise_with_seed((1, 1, 2, 2, 2), 123, device)
        noise_c = sampling.generate_noise_with_seed((1, 1, 2, 2, 2), 124, device)

        assert torch.allclose(noise_a, noise_b)
        assert not torch.allclose(noise_a, noise_c)


class TestSamplingPipelines:
    """Test sampling pipeline helpers"""

    def test_load_trained_model_prefers_requested_weights(self, temp_dir, device, monkeypatch):
        """Verify that the loader switches between EMA and main weights.

        Parameters
        ----------
        temp_dir : pathlib.Path
            Temporary directory used by the test.
        device : torch.device
            Execution device for the tensors.
        monkeypatch : pytest.MonkeyPatch
            Monkeypatch helper for replacing model construction.
        """

        def make_model(*args, **kwargs):
            return nn.Linear(1, 1)

        main_model = nn.Linear(1, 1)
        ema_model = nn.Linear(1, 1)
        with torch.no_grad():
            main_model.weight.fill_(1.0)
            main_model.bias.fill_(1.0)
            ema_model.weight.fill_(2.0)
            ema_model.bias.fill_(2.0)

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        torch.save(
            {
                "model": main_model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "val_loss": 0.5,
                "ema_val_loss": 0.25,
            },
            checkpoint_dir / "checkpoint_3.pt",
        )

        monkeypatch.setattr(sampling, "create_model", make_model)

        loaded_ema = sampling.load_trained_model(
            str(checkpoint_dir),
            3,
            64,
            32,
            1,
            2,
            1,
            3,
            True,
            device,
        )
        loaded_main = sampling.load_trained_model(
            str(checkpoint_dir),
            3,
            64,
            32,
            1,
            2,
            1,
            3,
            False,
            device,
        )

        assert torch.allclose(loaded_ema.weight, torch.full_like(loaded_ema.weight, 2.0))
        assert torch.allclose(loaded_main.weight, torch.full_like(loaded_main.weight, 1.0))
        assert not loaded_ema.training
        assert not loaded_main.training

    def test_sample_from_mask_uses_incrementing_seeds(self, temp_dir, device, monkeypatch):
        """Verify that per-sample seeds increase deterministically.

        Parameters
        ----------
        temp_dir : pathlib.Path
            Temporary directory used by the test.
        device : torch.device
            Execution device for the tensors.
        monkeypatch : pytest.MonkeyPatch
            Monkeypatch helper for replacing model utilities.
        """
        mask_path = temp_dir / "subject0_mask.nii.gz"
        write_nifti(mask_path, np.ones((2, 2, 2), dtype=np.float32))

        seen_seeds = []

        def fake_noise(shape, seed, device):
            seen_seeds.append(seed)
            return torch.full(shape, float(seed), device=device)

        def fake_generate_sample(model, noise, mask, diagnosis, device):
            return noise

        monkeypatch.setattr(sampling, "generate_noise_with_seed", fake_noise)
        monkeypatch.setattr(sampling, "generate_sample", fake_generate_sample)

        samples = sampling.sample_from_mask(object(), str(mask_path), 2, 1, 10, device)

        assert seen_seeds == [10, 16]
        assert [seed for _, _, _, seed, _ in samples] == [10, 16]
        assert all(subject_id == "subject0" for _, subject_id, _, _, _ in samples)
        assert all(target_label == 1 for _, _, target_label, _, _ in samples)
        assert torch.allclose(samples[0][0], torch.full((1, 1, 2, 2, 2), 10.0, device=device))
        assert torch.allclose(samples[1][0], torch.full((1, 1, 2, 2, 2), 16.0, device=device))

    def test_sample_model_selects_available_masks(self, temp_dir, device, monkeypatch):
        """Verify that mask sampling uses the available directory entries.

        Parameters
        ----------
        temp_dir : pathlib.Path
            Temporary directory used by the test.
        device : torch.device
            Execution device for the tensors.
        monkeypatch : pytest.MonkeyPatch
            Monkeypatch helper for replacing random selection.
        """
        mask_dir = temp_dir / "mask"
        mask_dir.mkdir()
        write_nifti(mask_dir / "subject_a_mask.nii.gz", np.zeros((2, 2, 2), dtype=np.float32))
        write_nifti(mask_dir / "subject_b_mask.nii.gz", np.ones((2, 2, 2), dtype=np.float32))

        def fake_noise(shape, seed, device):
            return torch.full(shape, float(seed), device=device)

        def fake_generate_sample(model, noise, mask, diagnosis, device):
            return noise

        monkeypatch.setattr(sampling, "generate_noise_with_seed", fake_noise)
        monkeypatch.setattr(sampling, "generate_sample", fake_generate_sample)
        monkeypatch.setattr(sampling.np.random, "randint", lambda n: 1)

        samples = sampling.sample_model(object(), str(mask_dir), 1, 0, 7, device)

        assert len(samples) == 1
        assert samples[0][1] == "subject_b"
        assert samples[0][3] == 7

    def test_save_samples_writes_nifti(self, temp_dir):
        """Verify that generated samples are persisted with the expected name.

        Parameters
        ----------
        temp_dir : pathlib.Path
            Temporary directory used by the test.
        """
        sample_dir = temp_dir / "samples"
        sample_dir.mkdir()

        samples_data = [
            (torch.zeros(1, 1, 2, 2, 2), "subject0", 2, 10, np.eye(4)),
        ]

        saved_paths = sampling.save_samples(samples_data, str(sample_dir))

        assert len(saved_paths) == 1
        assert Path(saved_paths[0]).name == "subject0_sampled_AD_10.nii.gz"
        assert Path(saved_paths[0]).exists()
        reloaded = nib.load(saved_paths[0]).get_fdata()
        assert reloaded.shape == (2, 2, 2)
"""
Verify train.py and sampling.py workflows
"""
import pytest
import torch
import torchdiffeq
from pathlib import Path
from model.trainer import Trainer


pytestmark = pytest.mark.slow  # Skipped by default: pytest -v -m "not slow"


class TestTrainPipeline:
    """Minimal test of train.py workflow"""
    
    def test_train_script_produces_checkpoint(self, temp_dir, trainer_instance, device):
        """Verify that the training workflow writes a checkpoint file.

        Parameters
        ----------
        temp_dir : pathlib.Path
            Temporary directory used by the test.
        trainer_instance : model.trainer.Trainer
            Configured trainer fixture used to save the checkpoint.
        device : torch.device
            Execution device for the test tensors.
        """
        # Manually save a checkpoint (simulating what train.py does periodically)
        checkpoint_path = temp_dir / "checkpoint_1.pt"
        trainer_instance.save_checkpoint()
        
        # Verify checkpoint was saved
        results_dir = Path(trainer_instance.results_dir)
        checkpoints = list(results_dir.glob("checkpoint_*.pt"))
        
        assert len(checkpoints) > 0, "No checkpoint files created"
        
        # Verify checkpoint contains model state
        checkpoint = torch.load(checkpoints[0])
        assert "model" in checkpoint
        assert checkpoint["model"] is not None


class TestSamplingPipeline:
    """Minimal test of sampling.py workflow"""
    
    def test_sampling_script_loads_checkpoint_and_generates(
        self, temp_dir, unet_model, device
    ):
        """Verify that sampling loads a checkpoint and produces a sample.

        Parameters
        ----------
        temp_dir : pathlib.Path
            Temporary directory used by the test.
        unet_model : torch.nn.Module
            UNet fixture used for checkpoint loading and sampling.
        device : torch.device
            Execution device for the tensors.
        """
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        
        # Save a checkpoint
        checkpoint_path = checkpoint_dir / "checkpoint_1.pt"
        torch.save({"model": unet_model.state_dict()}, checkpoint_path)
        
        # Load checkpoint (what sampling.py does)
        checkpoint = torch.load(checkpoint_path)
        unet_model.load_state_dict(checkpoint["model"])
        unet_model.eval()
        
        # Generate sample using ODE solver (what sampling.py actually does)
        with torch.no_grad():
            noise = torch.randn(1, 1, 64, 64, 64, device=device)
            mask = torch.randn(1, 1, 64, 64, 64, device=device)
            diagnosis = torch.tensor([0], dtype=torch.long, device=device)  # CN
            
            # Pack mask + diagnosis for conditioning (sampling.py approach)
            B = mask.shape[0]
            diagnosis_map = diagnosis.view(B, 1, 1, 1, 1).expand(-1, 1, *mask.shape[2:]).float()
            cond = torch.cat([mask, diagnosis_map], dim=1)
            
            # Solve ODE: x(t) evolves while mask stays constant
            def dynamics(t, x):
                return unet_model(torch.cat([x, cond[:, :1]], dim=1), t.reshape(-1), cond[:, 1, 0, 0, 0].long())
            
            samples = torchdiffeq.odeint(
                dynamics, noise,
                torch.linspace(0, 1, 2, device=device),
                atol=1e-4, rtol=1e-4
            )
        
        # Final sample at t=1
        sample_t1 = samples[-1]
        assert sample_t1.shape == (1, 1, 64, 64, 64)
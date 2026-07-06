"""Tests for UNet model"""

import pytest
import torch
import tempfile
import os
from model.unet_ADNI import create_model


class TestModelCreation:
    """Test model creation"""
    
    def test_create_model(self, mock_config):
        """Verify that the configured UNet model can be created.

        Parameters
        ----------
        mock_config : dict
            Mock configuration used to build the model.
        """
        params = mock_config["parameters"]
        model = create_model(
            image_size=params["input_size"],
            num_channels=params["num_channels"],
            num_res_blocks=params["num_res_blocks"],
            in_channels=params["in_channels"],
            out_channels=params["out_channels"],
            num_classes=params["num_classes"],
            class_cond=True,
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_create_model_without_conditioning(self):
        """Verify that the model can be created without class conditioning.

        Returns
        -------
        None
        """
        model = create_model(
            image_size=128,
            num_channels=64,
            num_res_blocks=2,
            in_channels=1,
            out_channels=1,
            num_classes=None,
            class_cond=False,
        )
        assert model is not None
        assert model.num_classes is None


class TestModelForwardPass:
    """Test model forward pass"""
    
    def test_forward_pass(self, unet_model, device):
        """Verify the forward pass output shape for a batched input.

        Parameters
        ----------
        unet_model : torch.nn.Module
            The test UNet model fixture.
        device : torch.device
            Execution device for the tensors.
        """
        batch_size = 2
        input_tensor = torch.randn(batch_size, 2, 64, 64, 64, device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        diagnosis = torch.randint(0, 3, (batch_size,), device=device)
        
        with torch.no_grad():
            output = unet_model(input_tensor, timesteps, diagnosis)
        
        assert output.shape == (batch_size, 1, 64, 64, 64)
        assert output.device == device
    
    def test_model_eval_mode(self, unet_model, device):
        """Verify that the model runs correctly in evaluation mode.

        Parameters
        ----------
        unet_model : torch.nn.Module
            The test UNet model fixture.
        device : torch.device
            Execution device for the tensors.
        """
        unet_model.eval()
        
        x = torch.randn(1, 2, 64, 64, 64, device=device)
        t = torch.tensor([500], device=device)
        c = torch.tensor([0], device=device)
        
        with torch.no_grad():
            output = unet_model(x, t, c)
        
        assert output is not None
        assert output.shape[1] == 1  # Output channels
    
    def test_training_mode(self, unet_model, device):
        """Verify that the model runs correctly in training mode.

        Parameters
        ----------
        unet_model : torch.nn.Module
            The test UNet model fixture.
        device : torch.device
            Execution device for the tensors.
        """
        unet_model.train()
        
        x = torch.randn(1, 2, 64, 64, 64, device=device)
        t = torch.tensor([500], device=device)
        c = torch.tensor([0], device=device)
        
        output = unet_model(x, t, c)
        
        assert output is not None
        assert output.shape == (1, 1, 64, 64, 64)


class TestModelProperties:
    """Test model properties"""
    
    def test_model_has_parameters(self, unet_model):
        """Verify that the model exposes trainable parameters.

        Parameters
        ----------
        unet_model : torch.nn.Module
            The test UNet model fixture.
        """
        params = list(unet_model.parameters())
        assert len(params) > 0
    
    def test_model_state_dict(self):
        """Verify that the model state dict can be saved and reloaded.

        Returns
        -------
        None
        """
        model = create_model(
            image_size=128,
            num_channels=64,
            num_res_blocks=2,
            in_channels=1,
            out_channels=1,
            num_classes=None,
            class_cond=False,
        )
        state = model.state_dict()
        
        # Use proper tempfile handling for cross-platform compatibility
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            torch.save(state, temp_path)
            loaded_state = torch.load(temp_path)
            assert loaded_state.keys() == state.keys()
        finally:
            # Ensure file is closed before deletion on Windows
            if os.path.exists(temp_path):
                os.remove(temp_path)

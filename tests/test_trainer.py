"""Tests for training and sampling"""

import pytest
import torch
from pathlib import Path
from model.trainer import perturb_mask
from .conftest import MockDataLoader


class TestTrainer:
    """Test Trainer class"""
    
    def test_trainer_init(self, trainer_instance, device):
        """Test trainer initialization"""
        assert trainer_instance is not None
        assert trainer_instance.model is not None
        assert trainer_instance.loader is not None
        assert trainer_instance.device == device
        assert trainer_instance.step == 0
    
    def test_compute_loss(self, trainer_instance, device):
        """Test loss computation"""
        # Create synthetic tensors for noise and velocity
        ut = torch.randn(1, 1, 64, 64, 64, device=device)
        vt = torch.randn(1, 1, 64, 64, 64, device=device)
        
        # Compute loss
        loss = trainer_instance.compute_loss(ut, vt)
        
        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
    
    def test_checkpoint_save(self, trainer_instance, temp_dir):
        """Test checkpoint saving"""
        trainer_instance.results_dir = str(temp_dir / "results")
        (temp_dir / "results").mkdir(exist_ok=True)
        
        trainer_instance.save_checkpoint(avg_loss=0.5)
        
        checkpoint_files = list(Path(trainer_instance.results_dir).glob("checkpoint_*.pt"))
        assert len(checkpoint_files) > 0
    
    def test_checkpoint_load(self, trainer_instance, temp_dir, device):
        """Test checkpoint loading"""
        trainer_instance.results_dir = str(temp_dir / "results")
        (temp_dir / "results").mkdir(exist_ok=True)
        
        # Save checkpoint
        trainer_instance.save_checkpoint(avg_loss=0.5)
        
        # Find checkpoint file
        checkpoint_files = list(Path(trainer_instance.results_dir).glob("checkpoint_*.pt"))
        assert len(checkpoint_files) > 0
        
        # Load and verify
        checkpoint_path = checkpoint_files[0]
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        assert "step" in checkpoint
        assert "model" in checkpoint
        assert isinstance(checkpoint["step"], int)
    
    def test_ema_update(self, trainer_instance):
        """Test EMA update"""
        if trainer_instance.use_ema:
            # Check EMA model exists
            assert trainer_instance.ema_model is not None
            
            # Call update
            trainer_instance.update_ema()
            
            # EMA model should have state dict
            ema_state = trainer_instance.ema_model.state_dict()
            assert len(ema_state) > 0


class TestLosses:
    """Test loss computation"""
    
    def test_loss_l1(self, trainer_instance, device):
        """Test L1 loss"""
        trainer_instance.loss_type = "l1"
        
        ut = torch.randn(1, 1, 64, 64, 64, device=device)
        vt = torch.randn(1, 1, 64, 64, 64, device=device)
        
        loss = trainer_instance.compute_loss(ut, vt)
        
        assert loss.item() >= 0
        assert loss.dim() == 0
    
    def test_loss_l2(self, trainer_instance, device):
        """Test L2 loss"""
        trainer_instance.loss_type = "l2"
        
        ut = torch.randn(1, 1, 64, 64, 64, device=device)
        vt = torch.randn(1, 1, 64, 64, 64, device=device)
        
        loss = trainer_instance.compute_loss(ut, vt)
        
        assert loss.item() >= 0
        assert loss.dim() == 0
    
    def test_loss_leh(self, trainer_instance, device):
        """Test LEH loss"""
        trainer_instance.loss_type = "leh"
        
        ut = torch.randn(1, 1, 64, 64, 64, device=device)
        vt = torch.randn(1, 1, 64, 64, 64, device=device)
        
        loss = trainer_instance.compute_loss(ut, vt)
        
        assert loss.item() >= 0
        assert loss.dim() == 0
    
    def test_loss_leb(self, trainer_instance, device):
        """Test LEB loss"""
        trainer_instance.loss_type = "leb"
        
        ut = torch.randn(1, 1, 64, 64, 64, device=device)
        vt = torch.randn(1, 1, 64, 64, 64, device=device)
        
        loss = trainer_instance.compute_loss(ut, vt)
        
        assert loss.item() >= 0
        assert loss.dim() == 0
    
    def test_loss_gradient_flow(self, trainer_instance, device):
        """Test that gradients flow through loss"""
        ut = torch.randn(1, 1, 64, 64, 64, device=device, requires_grad=True)
        vt = torch.randn(1, 1, 64, 64, 64, device=device)
        
        loss = trainer_instance.compute_loss(ut, vt)
        
        # Loss should be differentiable
        assert loss.requires_grad


class TestSampling:
    """Test sampling pipeline"""
    
    @pytest.mark.slow
    def test_sampling_produces_output(self, unet_model_eval, device):
        """Test sampling produces correct output shape"""
        noise = torch.randn(1, 1, 64, 64, 64, device=device)
        mask = torch.randn(1, 1, 64, 64, 64, device=device)
        diagnosis = torch.tensor([0], device=device)
        
        combined = torch.cat([noise, mask], dim=1)
        timestep = torch.tensor([500], device=device)
        
        with torch.no_grad():
            output = unet_model_eval(combined, timestep, diagnosis)
        
        assert output.shape == noise.shape
        assert output.device == device
    
    @pytest.mark.slow
    def test_sampling_all_diagnoses(self, unet_model_eval, device):
        """Test sampling for all diagnosis classes"""
        noise = torch.randn(1, 1, 64, 64, 64, device=device)
        mask = torch.randn(1, 1, 64, 64, 64, device=device)
        
        for diag in range(3):  # CN, MCI, AD
            diagnosis = torch.tensor([diag], device=device)
            combined = torch.cat([noise, mask], dim=1)
            timestep = torch.tensor([500], device=device)
            
            with torch.no_grad():
                output = unet_model_eval(combined, timestep, diagnosis)
            
            assert output is not None
            assert output.shape == noise.shape


class TestPipelines:
    """Test full pipelines"""
    
    @pytest.mark.slow
    def test_batch_consistency(self, unet_model_eval, device):
        """Test batch processing is consistent"""
        # Process as batch
        combined = torch.randn(2, 2, 64, 64, 64, device=device)
        timesteps = torch.tensor([500, 500], device=device)
        diagnosis = torch.tensor([0, 1], device=device)
        
        with torch.no_grad():
            batch_output = unet_model_eval(combined, timesteps, diagnosis)
        
        assert batch_output.shape == (2, 1, 64, 64, 64)


class TestValidation:
    """Test validation workflow"""
    
    def test_validate_with_loader(self, trainer_instance, device):
        """Test validation step"""
        if trainer_instance.val_loader is not None:
            # Create simple val loader
            val_batch = {
                "image": torch.randn(2, 1, 64, 64, 64, device=device),
                "mask": torch.randn(2, 1, 64, 64, 64, device=device),
                "diagnosis": torch.randint(0, 3, (2,), device=device),
            }
            
            # Validate should not crash
            trainer_instance.model.eval()
            with torch.no_grad():
                timesteps = torch.randint(0, 1000, (2,), device=device)
                combined = torch.cat([val_batch["image"], val_batch["mask"]], dim=1)
                output = trainer_instance.model(combined, timesteps, val_batch["diagnosis"])
            
            assert output is not None
            assert output.shape[0] == 2


class TestMaskPerturbation:
    """Test mask augmentation"""
    
    def test_perturb_mask_shape(self, device):
        """Test perturb_mask preserves shape"""
        mask = torch.randn(4, 1, 64, 64, 64, device=device)
        
        perturbed = perturb_mask(mask, apply_prob=1.0)
        
        assert perturbed.shape == mask.shape
        assert perturbed.device == device
    
    def test_perturb_mask_range(self, device):
        """Test perturb_mask stays in original range"""
        mask = torch.rand(4, 1, 64, 64, 64, device=device) * 2 - 1  # Range [-1, 1]
        orig_min = mask.min()
        orig_max = mask.max()
        
        perturbed = perturb_mask(mask, apply_prob=1.0)
        
        # Perturbed should be within original range (range preservation feature)
        assert perturbed.min() >= orig_min - 0.1
        assert perturbed.max() <= orig_max + 0.1
    
    def test_perturb_mask_skip_probability(self, device):
        """Test perturb_mask skip with low probability"""
        mask = torch.ones(1, 1, 64, 64, 64, device=device) * 0.5
        
        perturbed = perturb_mask(mask, apply_prob=0.0)  # Never apply
        
        # Should be unchanged when apply_prob=0
        assert torch.allclose(perturbed, mask)


class TestBatchSizes:
    """Test with different batch sizes"""

    @pytest.mark.slow
    def test_forward_pass_batch_8(self, unet_model, device):
        """Test model with realistic batch size 8"""
        batch_size = 8
        input_tensor = torch.randn(batch_size, 2, 64, 64, 64, device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        diagnosis = torch.randint(0, 3, (batch_size,), device=device)
        
        with torch.no_grad():
            output = unet_model(input_tensor, timesteps, diagnosis)
        
        assert output.shape == (batch_size, 1, 64, 64, 64)
        assert output.device == device
    
    #@pytest.mark.slow
    def test_loss_computation_batch_4(self, trainer_instance, device):
        """Test loss computation with batch size 4"""
        ut = torch.randn(4, 1, 64, 64, 64, device=device)
        vt = torch.randn(4, 1, 64, 64, 64, device=device)
        
        loss = trainer_instance.compute_loss(ut, vt)
        
        assert loss.item() >= 0
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)


class TestTrainerSchedulers:
    """Test trainer with different learning rate schedulers"""
    
    def test_trainer_cosine_scheduler(self, device, mock_config_no_wandb, unet_model, dummy_loader_single):
        """Test trainer initialization with cosine annealing scheduler"""
        from model.trainer import Trainer
        
        trainer = Trainer(
            model=unet_model,
            loader=dummy_loader_single,
            val_loader=dummy_loader_single,
            device=device,
            batch_size=2,
            epochs=2,
            lr=2e-4,
            loss_type="leh",
            scheduler_type="cos",
            t_max_step=1000,
            results_dir="./test_results",
            wb_run=None
        )
        
        assert trainer.lr_scheduler is not None
        assert trainer.loss_type == "leh"
    
    def test_trainer_exponential_scheduler(self, device, mock_config_no_wandb, unet_model, dummy_loader_single):
        """Test trainer with exponential decay scheduler"""
        from model.trainer import Trainer
        
        trainer = Trainer(
            model=unet_model,
            loader=dummy_loader_single,
            val_loader=dummy_loader_single,
            device=device,
            batch_size=2,
            epochs=2,
            lr=2e-4,
            loss_type="l2",
            scheduler_type="exp",
            gamma_decay=0.999,
            results_dir="./test_results",
            wb_run=None
        )
        
        assert trainer.lr_scheduler is not None
    
    def test_trainer_no_scheduler(self, device, mock_config_no_wandb, unet_model, dummy_loader_single):
        """Test trainer with no learning rate scheduler"""
        from model.trainer import Trainer
        
        trainer = Trainer(
            model=unet_model,
            loader=dummy_loader_single,
            val_loader=dummy_loader_single,
            device=device,
            batch_size=2,
            epochs=2,
            lr=2e-4,
            loss_type="l1",
            scheduler_type=None,
            results_dir="./test_results",
            wb_run=None
        )
        
        assert trainer.lr_scheduler is None


class TestValidationAndEMA:
    """Test validation and EMA model functionality"""
    
    def test_validate_method_callable(self, trainer_instance, device):
        """Test that validate method exists and is callable"""
        assert hasattr(trainer_instance, 'validate')
        assert callable(trainer_instance.validate)
    
    def test_trainer_with_ema(self, device, mock_config_no_wandb, unet_model, dummy_loader_single):
        """Test trainer initialization with EMA model"""
        from model.trainer import Trainer
        
        trainer = Trainer(
            model=unet_model,
            loader=dummy_loader_single,
            val_loader=dummy_loader_single,
            device=device,
            batch_size=2,
            epochs=2,
            lr=2e-4,
            loss_type="leh",
            use_ema=True,
            ema_decay=0.9999,
            results_dir="./test_results",
            wb_run=None
        )
        
        assert trainer.ema_model is not None
        # EMA model should have same architecture as main model
        assert len(list(trainer.ema_model.parameters())) == len(list(trainer.model.parameters()))
    
    def test_ema_parameter_tracking(self, trainer_instance, device):
        """Test that EMA model is created with trainer"""
        if trainer_instance.ema_model is None:
            pytest.skip("EMA not enabled in trainer_instance fixture")
        
        # Both models should have same structure
        model_params = len(list(trainer_instance.model.parameters()))
        ema_params = len(list(trainer_instance.ema_model.parameters()))
        assert model_params == ema_params, "EMA and main model should have same number of parameters"


@pytest.mark.slow
def test_training_single_step(device, tiny_unet_model, minimal_trainer_config, temp_dir):
    """Test single training step executes full pipeline"""
    from model.trainer import Trainer
    
    loader = MockDataLoader(num_batches=1, batch_size=1, device=device)
    
    trainer = Trainer(
        model=tiny_unet_model,
        loader=loader,
        val_loader=None,
        device=device,
        results_dir=str(temp_dir / "results"),
        **minimal_trainer_config,
        wb_run=None,
    )
    
    initial_params = [p.clone() for p in tiny_unet_model.parameters()]
    trainer.train()
    
    assert any(
        not torch.allclose(p_init, p) 
        for p_init, p in zip(initial_params, tiny_unet_model.parameters())
    ), "Model weights should change after training"
    
    assert trainer.step > 0, "Training step should increment"


@pytest.mark.slow
def test_trainer_validation(device, tiny_unet_model, temp_dir):
    """Test validation methods with actual val_loader"""
    from model.trainer import Trainer
    
    loader = MockDataLoader(num_batches=1, batch_size=1, device=device)
    val_loader = MockDataLoader(num_batches=1, batch_size=1, device=device)
    
    trainer = Trainer(
        model=tiny_unet_model,
        loader=loader,
        val_loader=val_loader,  # Now with validation
        device=device,
        batch_size=2,
        epochs=2,
        lr=2e-4,
        loss_type="leh",
        results_dir="./test_results",
        wb_run=None
    )
    
    # Test validate()
    val_loss = trainer.validate()
    assert val_loss is not None
    assert isinstance(val_loss, float)
    
    # Test validate_ema()
    if trainer.use_ema:
        ema_val_loss = trainer.validate_ema()
        assert ema_val_loss is not None
        assert isinstance(ema_val_loss, float)

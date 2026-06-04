"""pytest fixtures for tests"""

import pytest
import torch
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # Add project root to sys.path for imports

from model.unet_ADNI import create_model


# ============================================================================
# HELPERS AND MOCKS
# ============================================================================

def mock_yaml_config():
    """Create mock config dict"""
    return {
        "parameters": {
            "epochs": 2,
            "batch_size": 4,
            "input_size": 64,
            "num_channels": 64,
            "num_res_blocks": 2,
            "in_channels": 2,
            "out_channels": 1,
            "num_classes": 3,
            "save_every": 100,
            "lr": 2e-4,
            "loss_type": "leh",
            "warmup_steps": 0,
            "lr_scheduler": "cos",
            "lr_min": 2e-7,
            "gamma_decay": 0.9999,
            "t_max_step": 1000,
            "update_ema_every": 10,
            "use_ema": True,
            "ema_decay": 0.995,
            "grad_norm": 1.0,
            "weight_decay": 1e-4,
            "val_seeds": [42, 135, 654],
        },
        "directories": {
            "results_dir": "./test_results",
            "checkpoints_dir": "./test_checkpoints",
        },
        "checkpoint": {"run": "test_run", "checkpoint": "1"},
        "wandb": {"key": None},
    }


class MockDataLoader:
    """Simple mock DataLoader for testing"""
    
    def __init__(self, num_batches=1, batch_size=4, device=None):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
    
    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                "image": torch.randn(self.batch_size, 1, 64, 64, 64, device=self.device),
                "mask": torch.randn(self.batch_size, 1, 64, 64, 64, device=self.device),
                "diagnosis": torch.randint(0, 3, (self.batch_size,), device=self.device),
            }
    
    def __len__(self):
        return self.num_batches


# ============================================================================
# FIXTURES 
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """CPU device by default"""
    return torch.device("cpu")


@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Mock configuration"""
    return mock_yaml_config()


@pytest.fixture
def mock_config_no_wandb(mock_config):
    """Config without wandb"""
    mock_config["wandb"]["key"] = None
    return mock_config


@pytest.fixture
def dummy_loader_single(device):
    """Mock DataLoader (1 batch)"""
    return MockDataLoader(num_batches=1, batch_size=4, device=device)


@pytest.fixture
def unet_model(device, mock_config):
    """Pre-created UNet model"""
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
    model = model.to(device)
    yield model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def unet_model_eval(unet_model):
    """UNet in eval mode"""
    unet_model.eval()
    return unet_model


@pytest.fixture
def trainer_instance(device, unet_model, dummy_loader_single, mock_config_no_wandb, temp_dir):
    """Minimal Trainer for testing"""
    from model.trainer import Trainer
    
    (temp_dir / "results").mkdir(exist_ok=True)
    
    trainer = Trainer(
        model=unet_model,
        loader=dummy_loader_single,
        val_loader=None,
        device=device,
        batch_size=4,
        epochs=2,
        lr=2e-4,
        save_every=100,
        results_dir=str(temp_dir / "results"),
        loss_type="leh",
        use_ema=True,
        ema_decay=0.995,
        update_ema_every=10,
        warmup_steps=0,
        scheduler_type="cos",
        lr_min=2e-7,
        gamma_decay=0.9999,
        t_max_step=1000,
        grad_norm=1.0,
        weight_decay=1e-4,
        val_seeds=[42],
        checkpoint_path=None,
        wb_run=None,
    )
    
    yield trainer
    del trainer


@pytest.fixture
def tiny_unet_model(device):
    """Minimal UNet model for integration tests (CPU-friendly)"""
    model = create_model(
        image_size=64,
        num_channels=32,      # reduced from 64
        num_res_blocks=1,     # reduced from 2
        in_channels=2,
        out_channels=1,
        num_classes=3,
        class_cond=True,
    ).to(device)
    yield model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def minimal_trainer_config():
    """Minimal trainer config for integration tests (no overhead)"""
    return {
        "batch_size": 1,
        "epochs": 1,
        "lr": 2e-4,
        "loss_type": "leh",
        "scheduler_type": "cos",
        "t_max_step": 1,
        "save_every": 999,      # don't save
        "use_ema": False,       # skip EMA
        "warmup_steps": 0,      # no warmup
        "grad_norm": 1.0,
        "weight_decay": 1e-4,
    }


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset torch seed for reproducibility"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

"""Tests for fp16 utility helpers"""

import torch
import torch.nn as nn

from model import fp16_util


class TinyPrecisionModel(nn.Module):
    """Small model used to exercise fp16 utilities."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    def convert_to_fp16(self):
        self.apply(fp16_util.convert_module_to_f16)

    def convert_to_fp32(self):
        self.apply(fp16_util.convert_module_to_f32)


class TestPrecisionHelpers:
    """Test precision conversion helpers"""

    def test_module_precision_round_trip(self):
        """Verify that module conversion helpers round-trip the dtype.

        Returns
        -------
        None
        """
        layer = nn.Conv3d(1, 1, kernel_size=1)

        fp16_util.convert_module_to_f16(layer)
        assert layer.weight.dtype == torch.float16
        assert layer.bias.dtype == torch.float16

        fp16_util.convert_module_to_f32(layer)
        assert layer.weight.dtype == torch.float32
        assert layer.bias.dtype == torch.float32

    def test_get_param_groups_and_master_params(self):
        """Verify that named parameters are grouped into master tensors.

        Returns
        -------
        None
        """
        model = TinyPrecisionModel()
        param_groups_and_shapes = fp16_util.get_param_groups_and_shapes(model.named_parameters())
        master_params = fp16_util.make_master_params(param_groups_and_shapes)

        assert len(master_params) == 2
        assert all(param.dtype == torch.float32 for param in master_params)
        assert master_params[0].shape == torch.Size([1])
        assert master_params[1].shape == torch.Size([1, 1])

    def test_master_params_sync_back_to_model(self):
        """Verify that master parameters can be copied back into the model.

        Returns
        -------
        None
        """
        model = TinyPrecisionModel()
        param_groups_and_shapes = fp16_util.get_param_groups_and_shapes(model.named_parameters())
        master_params = fp16_util.make_master_params(param_groups_and_shapes)

        with torch.no_grad():
            master_params[0].fill_(3.0)
            master_params[1].fill_(4.0)

        fp16_util.master_params_to_model_params(param_groups_and_shapes, master_params)

        assert torch.allclose(model.conv.weight, torch.full_like(model.conv.weight, 4.0))
        assert torch.allclose(model.conv.bias, torch.full_like(model.conv.bias, 3.0))

    def test_state_dict_to_master_params(self):
        """Verify that state dict tensors can be recovered as master params.

        Returns
        -------
        None
        """
        model = TinyPrecisionModel()
        state_dict = model.state_dict()

        master_params = fp16_util.state_dict_to_master_params(model, state_dict, use_fp16=False)

        assert torch.allclose(master_params[0], state_dict["conv.weight"])
        assert torch.allclose(master_params[1], state_dict["conv.bias"])


class TestMixedPrecisionTrainer:
    """Test mixed precision trainer behavior"""

    def test_fp16_initialization_converts_model_and_builds_master_params(self):
        """Verify that fp16 initialization converts the model and keeps fp32 masters.

        Returns
        -------
        None
        """
        model = TinyPrecisionModel()
        trainer = fp16_util.MixedPrecisionTrainer(model=model, use_fp16=True)

        assert model.conv.weight.dtype == torch.float16
        assert model.conv.bias.dtype == torch.float16
        assert len(trainer.master_params) == len(list(model.parameters()))
        assert all(param.dtype == torch.float32 for param in trainer.master_params)

    def test_zero_grad_helpers(self):
        """Verify that gradient clearing helpers remove accumulated gradients.

        Returns
        -------
        None
        """
        model = TinyPrecisionModel()
        loss = model(torch.ones(1, 1, 2, 2, 2)).sum()
        loss.backward()

        fp16_util.zero_grad(model.parameters())

        assert all(param.grad is not None for param in model.parameters())
        assert all(torch.count_nonzero(param.grad).item() == 0 for param in model.parameters())

        param_groups_and_shapes = fp16_util.get_param_groups_and_shapes(model.named_parameters())
        master_params = fp16_util.make_master_params(param_groups_and_shapes)
        for param in master_params:
            param.grad = torch.ones_like(param)

        fp16_util.zero_master_grads(master_params)
        assert all(param.grad is None for param in master_params)

    def test_normal_optimize_updates_parameters(self):
        """Verify that the trainer performs a normal optimizer step.

        Returns
        -------
        None
        """
        model = TinyPrecisionModel()
        trainer = fp16_util.MixedPrecisionTrainer(model=model, use_fp16=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_weight = model.conv.weight.detach().clone()

        loss = model(torch.ones(1, 1, 2, 2, 2)).sum()
        trainer.backward(loss)
        assert trainer.optimize(optimizer) is True

        assert not torch.allclose(model.conv.weight, initial_weight)

    def test_check_overflow(self):
        """Verify that overflow detection handles finite and non-finite inputs.

        Returns
        -------
        None
        """
        assert not fp16_util.check_overflow(1.0)
        assert fp16_util.check_overflow(float("inf"))
        assert fp16_util.check_overflow(float("-inf"))
        assert fp16_util.check_overflow(float("nan"))
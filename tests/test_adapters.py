import torch
import torch.nn as nn

from mineralsam.adapters import (
    C3k2AdapterWrapper,
    adapter_parameter_ids,
    freeze_for_adapters,
    install_conv_adapters,
    parameter_summary,
)


class Conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, inputs):
        return self.conv(inputs)


class C3k2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = Conv(16, 16)
        self.cv2 = Conv(16, 16)
        self.m = nn.ModuleList([nn.Identity()])
        self.c = 16
        self.f = -1
        self.np = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, inputs):
        return self.cv2(self.cv1(inputs))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([C3k2(), nn.Conv2d(16, 16, 1)])


def test_adapter_preserves_shape_and_starts_near_host_output():
    torch.manual_seed(7)
    model = Model()
    inputs = torch.randn(2, 16, 12, 12)
    original = model.model[0](inputs).detach()
    result = install_conv_adapters(model)
    adapted = model.model[0](inputs).detach()

    assert result["n_adapters"] == 1
    assert isinstance(model.model[0], C3k2AdapterWrapper)
    assert adapted.shape == original.shape
    assert torch.mean(torch.abs(adapted - original)) < 0.02


def test_adapter_only_mode_freezes_host_parameters():
    model = Model()
    install_conv_adapters(model)
    freeze_for_adapters(model)
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}

    assert trainable == adapter_parameter_ids(model)
    assert 0 < parameter_summary(model)["ratio"] < 1


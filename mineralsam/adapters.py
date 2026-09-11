"""Parallel convolutional Adapters for YOLO26 C3k2 blocks."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional


EXCLUDED_LAYER_KEYWORDS = ("detect", "segment", "pose", "obb", "classify", "concat")


class LightBottleneck(nn.Module):
    """Pointwise residual bottleneck used inside the Adapter branch."""

    def __init__(self, channels: int):
        super().__init__()
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.pointwise.weight, a=1.0)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.pointwise(self.activation(inputs))


class ParallelC3k2Adapter(nn.Module):
    """Lightweight split-concatenate-project residual branch.

    The final projection starts near zero. Consequently, wrapping a pretrained
    block initially changes its output only slightly while preserving gradient
    flow through the complete Adapter branch.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bottleneck_ratio: float = 0.25,
        residual_scale: float = 0.25,
        blocks: int = 1,
        min_hidden_channels: int = 8,
        output_init_std: float = 1e-3,
    ):
        super().__init__()
        hidden = max(min_hidden_channels, round(output_channels * bottleneck_ratio))
        hidden = min(int(hidden), int(output_channels))
        blocks = max(1, min(2, int(blocks)))
        self.hidden_channels = hidden
        self.block_count = blocks
        self.gate = nn.Parameter(torch.tensor(float(residual_scale)))
        self.stem = nn.Conv2d(input_channels, 2 * hidden, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList(LightBottleneck(hidden) for _ in range(blocks))
        self.fuse = nn.Conv2d((2 + blocks) * hidden, output_channels, kernel_size=1, bias=True)
        self.activation = nn.GELU()

        nn.init.kaiming_uniform_(self.stem.weight, a=1.0)
        nn.init.zeros_(self.stem.bias)
        nn.init.normal_(self.fuse.weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(self.fuse.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = list(self.stem(inputs).chunk(2, dim=1))
        for block in self.blocks:
            features.append(block(features[-1]))
        delta = self.fuse(self.activation(torch.cat(features, dim=1)))
        return delta * self.gate

    def initialize_stem_from(self, host: nn.Module) -> bool:
        """Copy the compatible leading slice of the host input projection."""
        source = _get_conv2d(getattr(host, "cv1", None))
        if source is None:
            return False
        with torch.no_grad():
            out_channels = min(self.stem.weight.shape[0], source.weight.shape[0])
            in_channels = min(self.stem.weight.shape[1], source.weight.shape[1])
            if out_channels < 1 or in_channels < 1:
                return False
            self.stem.weight[:out_channels, :in_channels].copy_(
                source.weight[:out_channels, :in_channels]
            )
            self.stem.bias.zero_()
        return True


class C3k2AdapterWrapper(nn.Module):
    """Fuse a frozen host block and a trainable Adapter as `F(X) + A(X)`."""

    def __init__(
        self,
        host: nn.Module,
        input_channels: int,
        output_channels: int,
        bottleneck_ratio: float,
        residual_scale: float,
        output_init_std: float,
    ):
        super().__init__()
        self.adapter = ParallelC3k2Adapter(
            input_channels=input_channels,
            output_channels=output_channels,
            bottleneck_ratio=bottleneck_ratio,
            residual_scale=residual_scale,
            blocks=len(getattr(host, "m", ())) or 1,
            output_init_std=output_init_std,
        )
        self.adapter.initialize_stem_from(host)
        self.host = host
        self._copy_ultralytics_metadata(host)

    def _copy_ultralytics_metadata(self, host: nn.Module) -> None:
        for attribute in ("i", "f", "np"):
            if hasattr(host, attribute):
                setattr(self, attribute, getattr(host, attribute))
        base_type = getattr(host, "type", host.__class__.__name__)
        self.type = f"{base_type}+MineralSAMAdapter"
        self.np = int(getattr(host, "np", 0)) + sum(p.numel() for p in self.adapter.parameters())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._fuse(self.host(inputs), inputs)

    def forward_split(self, inputs: torch.Tensor) -> torch.Tensor:
        host_output = (
            self.host.forward_split(inputs)
            if hasattr(self.host, "forward_split")
            else self.host(inputs)
        )
        return self._fuse(host_output, inputs)

    def _fuse(self, host_output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if not all(isinstance(item, torch.Tensor) and item.ndim == 4 for item in (host_output, inputs)):
            return host_output
        delta = self.adapter(inputs)
        if delta.shape[-2:] != host_output.shape[-2:]:
            delta = functional.interpolate(
                delta, size=host_output.shape[-2:], mode="bilinear", align_corners=False
            )
        if delta.shape[1] != host_output.shape[1]:
            raise RuntimeError("Adapter and host channel counts do not match")
        return host_output + delta


def install_conv_adapters(
    model: nn.Module,
    target_layers: Optional[Iterable[int] | str] = None,
    bottleneck_ratio: float | str | None = "dynamic",
    residual_scale: float = 0.25,
    output_init_std: float = 1e-3,
) -> dict:
    """Insert an Adapter beside every eligible single-input C3k2 layer."""
    layers = getattr(model, "model", None)
    if layers is None or not hasattr(layers, "__len__"):
        raise ValueError("The model must expose an indexable `.model` layer sequence")
    targets = _normalize_targets(target_layers)
    details = []
    for index, layer in enumerate(list(layers)):
        if isinstance(layer, C3k2AdapterWrapper):
            continue
        if targets is not None and index not in targets:
            continue
        if not _is_c3k2(layer) or getattr(layer, "f", -1) != -1 or _is_excluded(layer):
            continue
        input_channels = _infer_input_channels(layer)
        output_channels = _infer_output_channels(layer)
        if input_channels is None or output_channels is None:
            continue
        ratio = _resolve_ratio(layer, bottleneck_ratio)
        wrapper = C3k2AdapterWrapper(
            layer,
            input_channels,
            output_channels,
            bottleneck_ratio=ratio,
            residual_scale=residual_scale,
            output_init_std=output_init_std,
        )
        _move_like(wrapper, layer)
        layers[index] = wrapper
        parameter_count = sum(p.numel() for p in wrapper.adapter.parameters())
        details.append(
            {
                "index": index,
                "input_channels": input_channels,
                "output_channels": output_channels,
                "hidden_channels": wrapper.adapter.hidden_channels,
                "blocks": wrapper.adapter.block_count,
                "bottleneck_ratio": ratio,
                "parameters": parameter_count,
            }
        )
    return {
        "n_adapters": len(details),
        "adapter_parameters": sum(item["parameters"] for item in details),
        "details": details,
    }


def freeze_for_adapters(model: nn.Module) -> None:
    """Freeze the host network and leave only Adapter parameters trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, ParallelC3k2Adapter):
            for parameter in module.parameters():
                parameter.requires_grad = True


def adapter_parameter_ids(model: nn.Module) -> set[int]:
    return {
        id(parameter)
        for module in model.modules()
        if isinstance(module, ParallelC3k2Adapter)
        for parameter in module.parameters()
    }


def set_segment_head_trainable(model: nn.Module) -> set[int]:
    """Optionally unfreeze the prototype and mask-coefficient branches."""
    layers = getattr(model, "model", None)
    if layers is None or not len(layers):
        return set()
    head = layers[-1]
    if not hasattr(head, "proto"):
        return set()
    parameter_ids = set()
    for branch_name in ("proto", "cv4", "one2one_cv4"):
        branch = getattr(head, branch_name, None)
        if branch is None:
            continue
        for parameter in branch.parameters():
            parameter.requires_grad = True
            parameter_ids.add(id(parameter))
    return parameter_ids


def set_batchnorm_eval(model: nn.Module) -> int:
    """Keep BatchNorm running statistics fixed during few-shot adaptation."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def parameter_summary(model: nn.Module) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "ratio": trainable / total if total else 0.0}


def _normalize_targets(target_layers) -> Optional[set[int]]:
    if target_layers is None:
        return None
    if isinstance(target_layers, str):
        text = target_layers.strip().lower()
        if text in {"", "auto", "all", "c3k2", "all_c3k2"}:
            return None
        return {int(value.strip()) for value in text.split(",") if value.strip()}
    values = {int(value) for value in target_layers}
    return values or None


def _resolve_ratio(layer: nn.Module, value) -> float:
    if value is None or (isinstance(value, str) and value.lower() in {"auto", "dynamic", "paper"}):
        parameters = int(getattr(layer, "np", 0) or sum(p.numel() for p in layer.parameters()))
        if parameters > 5_000_000:
            return 1.0
        if parameters > 1_000_000:
            return 0.5
        return 0.25
    return max(0.01, float(value))


def _is_c3k2(layer: nn.Module) -> bool:
    return (
        layer.__class__.__name__ == "C3k2"
        and hasattr(layer, "cv1")
        and hasattr(layer, "cv2")
        and hasattr(layer, "m")
        and hasattr(layer, "c")
    )


def _is_excluded(layer: nn.Module) -> bool:
    names = (layer.__class__.__name__.lower(), str(getattr(layer, "type", "")).lower())
    return any(keyword in name for name in names for keyword in EXCLUDED_LAYER_KEYWORDS)


def _get_conv2d(module) -> Optional[nn.Conv2d]:
    if isinstance(module, nn.Conv2d):
        return module
    nested = getattr(module, "conv", None)
    return nested if isinstance(nested, nn.Conv2d) else None


def _infer_input_channels(layer: nn.Module) -> Optional[int]:
    for name in ("cv1", "conv"):
        convolution = _get_conv2d(getattr(layer, name, None))
        if convolution is not None:
            return int(convolution.in_channels)
    convolutions = [module for module in layer.modules() if isinstance(module, nn.Conv2d)]
    return int(convolutions[0].in_channels) if convolutions else None


def _infer_output_channels(layer: nn.Module) -> Optional[int]:
    for name in ("cv3", "cv2", "conv", "cv1"):
        convolution = _get_conv2d(getattr(layer, name, None))
        if convolution is not None:
            return int(convolution.out_channels)
    convolutions = [module for module in layer.modules() if isinstance(module, nn.Conv2d)]
    return int(convolutions[-1].out_channels) if convolutions else None


def _move_like(wrapper: nn.Module, host: nn.Module) -> None:
    reference = next(host.parameters(), None)
    if reference is None:
        return
    wrapper.to(device=reference.device)
    if reference.is_floating_point():
        wrapper.to(dtype=reference.dtype)


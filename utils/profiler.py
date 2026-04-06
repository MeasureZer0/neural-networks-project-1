from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

TensorDict = Dict[str, torch.Tensor]
ModelInput = Union[torch.Tensor, TensorDict]
ForwardInput = Tuple[torch.Tensor, TensorDict]


def model_stats(model: nn.Module, use_fp16: bool = False) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    bytes_per_param = 2 if use_fp16 else 4
    param_mb = total * bytes_per_param / 1024**2

    return {
        "total_param_M": total / 1e6,
        "trainable_param_M": trainable / 1e6,
        "frozen_param_M": frozen / 1e6,
        "param_size_MB": param_mb,
        "precision": 16.0 if use_fp16 else 32.0,
    }


def flop_stats(model: nn.Module, sample_input: ModelInput) -> Dict[str, float]:
    model.eval()

    inputs: Tuple[torch.Tensor, ...]

    if isinstance(sample_input, torch.Tensor):
        inputs = (sample_input[:1],)

    elif isinstance(sample_input, dict):
        inputs = tuple(v[:1] for v in sample_input.values())

    elif isinstance(sample_input, tuple):
        inputs = tuple(v[:1] for v in sample_input if isinstance(v, torch.Tensor))

    else:
        raise TypeError(f"Unsupported input type: {type(sample_input)}")

    flops = FlopCountAnalysis(model, inputs)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)

    return {"flops_G": flops.total() / 1e9}


def vram_stats(
    model: nn.Module,
    device: torch.device,
    sample_input: Union[ModelInput, ForwardInput],
    use_fp16: bool = False,
) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    stats = {}

    if use_fp16:
        model = model.half()

    torch.cuda.reset_peak_memory_stats()
    model.to(device)
    stats["vram_model_MB"] = torch.cuda.memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_fp16):
            if isinstance(sample_input, tuple):
                image, token_dict = sample_input
                args: ForwardInput = (
                    image[:1].to(device),
                    {k: v[:1].to(device) for k, v in token_dict.items()},
                )
                _ = model(*args)
            elif isinstance(sample_input, torch.Tensor):
                _ = model(sample_input[:1].to(device))
            else:
                x = {k: v[:1].to(device) for k, v in sample_input.items()}
                _ = model(x)

    stats["vram_forward_peak_MB"] = torch.cuda.max_memory_allocated() / 1024**2

    return stats


def print_stats(name: str, stats: Dict[str, float]) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:<30} {value:.2f}")
        else:
            print(f"  {key:<30} {value}")

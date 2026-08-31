import typing
from functools import partial
from typing import Optional, List
from torch import nn
import torch
from torch.utils.checkpoint import checkpoint
import torchtune
import torchtune.modules.peft
from torchtune.modules import TransformerSelfAttentionLayer


def checkpoint_wrap(
    model: nn.Module,
    param_threshold: Optional[int] = None,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None,
) -> List[str]:
    """
    Applies gradient checkpointing to leaf modules in a model.

    This function modifies the model in-place by wrapping leaf modules (modules
    without children) with gradient checkpointing. By default, all leaf modules
    are wrapped unless filtered by whitelist/blacklist parameters.

    Args:
        model (nn.Module): The model to be modified.
        param_threshold (Optional[int]): The minimum number of parameters a module must
            have to be considered for checkpointing. If None, no threshold is applied.
        whitelist (Optional[List[str]]): List of module name patterns to include.
            If provided, only modules whose names match any pattern in this list
            will be wrapped. Supports substring matching.
        blacklist (Optional[List[str]]): List of module name patterns to exclude.
            Modules whose names match any pattern in this list will not be wrapped.
            Blacklist takes precedence over whitelist. Supports substring matching.

    Returns:
        List[str]: List of names of wrapped modules.
    """
    wrapped_modules = []

    def _should_wrap_module(name: str) -> bool:
        """Check if a module should be wrapped based on whitelist/blacklist."""
        # Check blacklist first (takes precedence)
        if blacklist:
            for pattern in blacklist:
                if pattern in name:
                    return False

        # Check whitelist if provided
        if whitelist:
            for pattern in whitelist:
                if pattern in name:
                    return True
            return False  # If whitelist is provided but no match, don't wrap

        return True  # Default: wrap all modules if no whitelist/blacklist

    for name, module in model.named_modules():
        # Skip the root module
        if not name.strip():
            continue

        # Only wrap leaf modules (modules without children)
        has_children = any(module.children())
        if has_children:
            continue

        # Check whitelist/blacklist filtering
        if not _should_wrap_module(name):
            continue

        # Check parameter threshold if specified
        if param_threshold is not None:
            num_params = sum(p.numel() for p in module.parameters())
            if num_params < param_threshold:
                continue

        # Wrap the leaf module with gradient checkpointing
        module.forward = partial(checkpoint, module.forward, use_reentrant=False)
        wrapped_modules.append(name)

    return wrapped_modules


def trainable_parameters(m: torch.nn.TransformerDecoder) -> str:
    res = ""
    total_param_num = 0
    total_trainable_num = 0

    def handle_single_layer(k, v, indent) -> typing.Tuple[str, int, int]:
        r = ""
        layer_param_num = 0
        layer_trainable_num = 0
        for param in v.parameters():
            layer_param_num += param.numel()
            if param.requires_grad:
                layer_trainable_num += param.numel()
        mod_str = repr(v)
        indent_str = "\t" * indent
        r += f"{indent_str}({k}): {mod_str}, trainable parameters: {layer_trainable_num}/{layer_param_num}={layer_trainable_num/layer_param_num if layer_param_num > 0 else 0:.2f}\n"
        return r, layer_param_num, layer_trainable_num

    def handle_list_layer(
        k, v: torch.nn.ModuleList, indent
    ) -> typing.Tuple[str, int, int]:
        r = ""
        clz_num_d = dict()
        clz_repr_d = dict()
        param_num = trainable_param_num = 0
        mod_str = ""
        for sub_mod in v:
            if sub_mod.__class__.__name__ in clz_num_d:
                clz_num_d[sub_mod.__class__.__name__] += 1
            else:
                clz_num_d[sub_mod.__class__.__name__] = 1
                clz_repr_d[sub_mod.__class__.__name__] = ("", sub_mod)
        indent_str = "\t" * (indent + 1)
        for mod_clz, num in clz_num_d.items():
            sub_mod_str, sub_param, sub_trainable = handle_layer(
                clz_repr_d[mod_clz][0], clz_repr_d[mod_clz][1], indent + 1
            )
            sub_mod_str = sub_mod_str.strip("\t")
            param_num += sub_param
            trainable_param_num += sub_trainable
            mod_str += f"{indent_str}(0-{num - 1}): {num} x {sub_mod_str}\n"
        indent_str = "\t" * indent
        r = f"{indent_str}({k}): ModuleList(\n{mod_str}{indent_str}), trainable parameters: {trainable_param_num}/{param_num}={trainable_param_num/param_num}\n"
        return r, param_num, trainable_param_num

    def handle_layer(k, v, indent) -> typing.Tuple[str, int, int]:
        res = ""
        param_num = trainable_param_num = 0
        if isinstance(v, torch.nn.ModuleList):
            s, p, t = handle_list_layer(k, v, indent)
            res += s
            param_num += p
            trainable_param_num += t
        elif isinstance(
            v,
            (
                TransformerSelfAttentionLayer,
                torchtune.modules.transformer.TransformerDecoder,
                torchtune.modules.transformer.MultiHeadAttention,
                torchtune.modules.feed_forward.FeedForward,
                torchtune.modules.peft.LoRALinear,
            ),
        ) or v.__class__.__name__.__contains__("CheckpointWrapper"):
            sub_res = ""
            for sub_k, sub_mod in v._modules.items():
                s, p, t = handle_layer(sub_k, sub_mod, indent + 1)
                sub_res += s
                param_num += p
                trainable_param_num += t
            indent_str = "\t" * indent
            res += f"{indent_str}({k}): {v.__class__.__name__}(\n{sub_res}{indent_str}), trainable parameters: {trainable_param_num}/{param_num}={trainable_param_num/param_num}\n"
        else:
            s, p, t = handle_single_layer(k, v, indent)
            res += s
            param_num += p
            trainable_param_num += t
        return res, param_num, trainable_param_num

    s, p, t = handle_layer("", m, 0)
    res += s
    total_param_num += p
    total_trainable_num += t
    res += f"total param: {total_param_num}, trainable param: {total_trainable_num}, {total_trainable_num/total_param_num:.5f}"

    return res

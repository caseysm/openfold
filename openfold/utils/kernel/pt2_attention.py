# Copyright 2026 Megure Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch 2 capture support for OpenFold's fused attention softmax."""

import importlib
from typing import Callable, Dict

import torch
from torch.library import custom_op


def load_attn_core_extension():
    return importlib.import_module("attn_core_inplace_cuda")


def _native_attention_softmax(input: torch.Tensor) -> torch.Tensor:
    output = input.contiguous().clone()
    load_attn_core_extension().forward_(
        output,
        output.numel() // output.shape[-1],
        output.shape[-1],
    )
    return output


@custom_op("openfold::attention_softmax", mutates_args=())
def attention_softmax(input: torch.Tensor) -> torch.Tensor:
    """Functional dispatcher boundary around the in-place CUDA kernel."""
    return _native_attention_softmax(input)


@attention_softmax.register_fake
def _attention_softmax_fake(input: torch.Tensor) -> torch.Tensor:
    return input.new_empty(input.shape)


def _attention_softmax_setup_context(ctx, inputs, output):
    ctx.save_for_backward(output)


def _attention_softmax_backward(ctx, grad_output):
    output, = ctx.saved_tensors
    dot = torch.sum(grad_output * output, dim=-1, keepdim=True)
    return output * (grad_output - dot)


attention_softmax.register_autograd(
    _attention_softmax_backward,
    setup_context=_attention_softmax_setup_context,
)


def attention_softmax_decomposition(input: torch.Tensor) -> torch.Tensor:
    """Lower the capture-only operator to portable ATen operations."""
    return torch.softmax(input, dim=-1)


def attention_softmax_decomposition_table() -> Dict[object, Callable]:
    """Return the decomposition table expected by ExportedProgram."""
    return {
        torch.ops.openfold.attention_softmax.default:
            attention_softmax_decomposition,
    }


def _is_fake_tensor(input: torch.Tensor) -> bool:
    try:
        from torch._subclasses.fake_tensor import is_fake
    except ImportError:
        return False
    return is_fake(input)


def is_pt2_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    return bool(is_compiling is not None and is_compiling())


def attention_softmax_inplace(input: torch.Tensor) -> torch.Tensor:
    """Use the original in-place kernel eagerly and a functional op in PT2."""
    if is_pt2_compiling() or _is_fake_tensor(input) or input.device.type == "meta":
        return attention_softmax(input)

    load_attn_core_extension().forward_(
        input,
        input.numel() // input.shape[-1],
        input.shape[-1],
    )
    return input

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from unittest import mock

import torch

from openfold.utils.kernel.attention_core import attention_core
from openfold.utils.kernel.pt2_attention import (
    attention_softmax,
    attention_softmax_decomposition_table,
    attention_softmax_inplace,
)


class _SoftmaxModule(torch.nn.Module):
    def forward(self, input):
        return attention_softmax_inplace(input)


class _AttentionCoreModule(torch.nn.Module):
    def forward(self, q, k, v, bias):
        return attention_core(q, k, v, bias, None)


class _FakeNativeExtension:
    @staticmethod
    def forward_(input, rows, cols):
        if rows != input.numel() // input.shape[-1]:
            raise AssertionError("invalid row count")
        if cols != input.shape[-1]:
            raise AssertionError("invalid column count")
        input.copy_(torch.softmax(input, dim=-1))


def _targets(exported_program):
    return {node.target for node in exported_program.graph.nodes}


class TestPT2AttentionSoftmax(unittest.TestCase):
    def test_eager_path_preserves_inplace_kernel_behavior(self):
        input = torch.randn(2, 3, 5)
        expected = torch.softmax(input, dim=-1)

        with mock.patch(
            "openfold.utils.kernel.pt2_attention.load_attn_core_extension",
            return_value=_FakeNativeExtension(),
        ):
            output = attention_softmax_inplace(input)

        self.assertIs(output, input)
        torch.testing.assert_close(output, expected)

    def test_custom_op_autograd_matches_aten_softmax(self):
        input = torch.randn(2, 3, 5, requires_grad=True)
        input_reference = input.detach().clone().requires_grad_(True)
        weight = torch.randn_like(input)

        with mock.patch(
            "openfold.utils.kernel.pt2_attention.load_attn_core_extension",
            return_value=_FakeNativeExtension(),
        ):
            output = attention_softmax(input)
            torch.sum(output * weight).backward()

        output_reference = torch.softmax(input_reference, dim=-1)
        torch.sum(output_reference * weight).backward()

        torch.testing.assert_close(output, output_reference)
        torch.testing.assert_close(input.grad, input_reference.grad)

    def test_custom_op_has_fake_tensor_implementation(self):
        from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

        with FakeTensorMode() as mode:
            input = mode.from_tensor(torch.randn(2, 3, 5))
            output = attention_softmax(input)

        self.assertTrue(is_fake(output))
        self.assertEqual(output.shape, input.shape)
        self.assertEqual(output.dtype, input.dtype)

    def test_export_uses_custom_op_then_decomposes_to_aten(self):
        input = torch.randn(2, 3, 5)
        exported = torch.export.export(_SoftmaxModule(), (input,), strict=True)

        self.assertIn(
            torch.ops.openfold.attention_softmax.default,
            _targets(exported),
        )

        lowered = exported.run_decompositions(
            attention_softmax_decomposition_table()
        )
        self.assertNotIn(
            torch.ops.openfold.attention_softmax.default,
            _targets(lowered),
        )
        torch.testing.assert_close(
            lowered.module()(input),
            torch.softmax(input, dim=-1),
        )

    def test_attention_core_exports_without_loading_native_extension(self):
        q = torch.randn(2, 3, 5, 4)
        k = torch.randn(2, 3, 5, 4)
        v = torch.randn(2, 3, 5, 6)
        bias = torch.randn(2, 1, 1, 5)

        exported = torch.export.export(
            _AttentionCoreModule(),
            (q, k, v, bias),
            strict=True,
        )
        self.assertIn(
            torch.ops.openfold.attention_softmax.default,
            _targets(exported),
        )
        lowered = exported.run_decompositions(
            attention_softmax_decomposition_table()
        )
        self.assertNotIn(
            torch.ops.openfold.attention_softmax.default,
            _targets(lowered),
        )

        expected = torch.matmul(
            torch.softmax(torch.matmul(q, k.transpose(-1, -2)) + bias, dim=-1),
            v,
        )
        torch.testing.assert_close(
            lowered.module()(q, k, v, bias),
            expected,
        )


if __name__ == '__main__':
    unittest.main()

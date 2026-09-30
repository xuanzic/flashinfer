# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer.quantization.kernels import (
    nvfp4_kv_materialize_active_pages_qmul4,
)


def _swizzle_v_scales(linear: torch.Tensor) -> torch.Tensor:
    pages, heads, tokens, scale_dim = linear.shape
    return (
        linear.reshape(pages, heads, tokens // 4, 4, 4, scale_dim // 4)
        .permute(0, 1, 2, 4, 5, 3)
        .reshape(pages, heads, tokens, scale_dim)
        .contiguous()
    )


def _add_page_padding(tensor: torch.Tensor, padding: int = 64) -> torch.Tensor:
    """Copy a compact tensor into a view with padding between physical pages."""
    compact_page_stride = tensor.stride(0)
    padded_page_stride = compact_page_stride + padding
    storage_size = padded_page_stride * (tensor.shape[0] - 1) + compact_page_stride
    storage = torch.empty(storage_size, dtype=tensor.dtype, device=tensor.device)
    padded = torch.as_strided(
        storage,
        size=tensor.shape,
        stride=(padded_page_stride, *tensor.stride()[1:]),
    )
    padded.copy_(tensor)
    return padded


def _dequantize_reference(
    data: torch.Tensor,
    scales: torch.Tensor,
    page_ids: torch.Tensor,
) -> torch.Tensor:
    data = data.index_select(0, page_ids.to(torch.long))
    scales = scales.index_select(0, page_ids.to(torch.long))
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float16,
        device=data.device,
    )
    low = lut[(data & 0x0F).long()]
    high = lut[(data >> 4).long()]
    values = torch.stack((low, high), dim=-1).flatten(-2)
    expanded_scales = scales.to(torch.float8_e4m3fn).to(torch.float16)
    expanded_scales = expanded_scales.repeat_interleave(16, dim=-1)
    return (values * expanded_scales).to(torch.float8_e4m3fn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("page_strided", [False, True])
@pytest.mark.parametrize("provide_output", [False, True])
def test_nvfp4_active_page_materialize_matches_reference(
    head_dim: int,
    page_strided: bool,
    provide_output: bool,
) -> None:
    physical_pages = 7
    num_heads = 2
    page_size = 64
    scale_dim = head_dim // 16
    generator = torch.Generator(device="cuda").manual_seed(20260806 + head_dim)
    k_data = torch.randint(
        0,
        256,
        (physical_pages, num_heads, page_size, head_dim // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    v_data = torch.randint(
        0,
        256,
        k_data.shape,
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    k_scales = (
        torch.rand(
            (physical_pages, num_heads, page_size, scale_dim),
            device="cuda",
            generator=generator,
        )
        * 1.5
        + 0.01
    ).to(torch.float8_e4m3fn)
    v_scales_linear = (
        torch.rand(
            (physical_pages, num_heads, page_size, scale_dim),
            device="cuda",
            generator=generator,
        )
        * 1.5
        + 0.01
    ).to(torch.float8_e4m3fn)
    v_scales_physical = _swizzle_v_scales(v_scales_linear)
    page_ids = torch.tensor([3, 1, 0, 6, 2, 3], dtype=torch.int32, device="cuda")

    k_scales_u8 = k_scales.view(torch.uint8)
    v_scales_u8 = v_scales_physical.view(torch.uint8)
    if page_strided:
        k_data = _add_page_padding(k_data)
        v_data = _add_page_padding(v_data)
        k_scales_u8 = _add_page_padding(k_scales_u8)
        v_scales_u8 = _add_page_padding(v_scales_u8)

    output = None
    if provide_output:
        output = torch.empty(
            (page_ids.numel() + 1, 2, num_heads, page_size, head_dim),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )

    actual = nvfp4_kv_materialize_active_pages_qmul4(
        k_data,
        v_data,
        k_scales_u8,
        v_scales_u8,
        page_ids,
        output,
    )
    if output is not None:
        assert actual.data_ptr() == output.data_ptr()
    expected_k = _dequantize_reference(k_data, k_scales, page_ids)
    expected_v = _dequantize_reference(v_data, v_scales_linear, page_ids)
    torch.cuda.synchronize()

    actual_k, actual_v = actual.unbind(dim=1)
    torch.testing.assert_close(
        actual_k[1:].view(torch.uint8), expected_k.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_v[1:].view(torch.uint8), expected_v.view(torch.uint8), rtol=0, atol=0
    )

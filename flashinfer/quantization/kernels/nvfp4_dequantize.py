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

"""Standalone NVFP4-to-E4M3 dequantization using Blackwell QMUL4.

This development kernel intentionally applies only the NVFP4 per-block E4M3
scale.  It does not apply the tensor-level/global K or V scale.  Attention
callers must pass that scale to the downstream FP8 FMHA exactly once.

The kernel consumes the linear representation used by
``nvfp4_kv_dequantize``: packed E2M1 data in ``[M, K / 2]`` and one E4M3
scale byte per 16 logical elements in ``[M, K / 16]``.  It is M-agnostic and
is compiled once per logical head dimension K.
"""

import functools
import os
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int16, Int32, Uint8, Uint32

from ...api_logging import flashinfer_api
from ...cute_dsl.fp4_common import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    st_global_v4_u32,
)
from ...cute_dsl.utils import get_num_sm
from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel


_NVFP4_BLOCK_SIZE = 16
_THREADS_PER_BLOCK = 256
_BLOCKS_PER_SM = 4


def _kernel_source_files() -> tuple[str, ...]:
    """Source files whose content invalidates the on-disk kernel cache."""
    from ...cute_dsl import fp4_common

    return (__file__, fp4_common.__file__)


@cute.jit
def _broadcast_low_byte(value: Int32) -> Int32:
    """Broadcast the low byte of ``value`` into four E4M3 scale lanes."""
    return cute.arch.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, {$r0}, 0x0000;",
        write_only_types=[Int32],
        read_only_args=[value],
    )


@cute.jit
def _mul_packed_e2m1x4_e4m3x4(packed_fp4: Int16, packed_sf: Int32) -> Int32:
    """Apply four E4M3 scales to four contiguous packed E2M1 nibbles."""
    if cutlass.const_expr(cutlass.target_version(min_version="13.4")):
        return cute.arch.inline_ptx(
            "mul.e4m3x4.e2m1x4.e4m3x4.satfinite {$w0}, {$r0}, {$r1};",
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )
    else:
        # Portable control/fallback path for toolchains predating PTX 9.4.
        # Unlike the fused decode helper, this input already stores four
        # consecutive nibbles in bits 0/4/8/12, so no LDSM compaction is
        # required.
        return cute.arch.inline_ptx(
            """
            {
                .reg .b8 fp4_01, fp4_23;
                .reg .b16 sf_01, sf_23, e4m3_01, e4m3_23;
                .reg .b32 h_01, h_23, sf_h_01, sf_h_23;
                mov.b16 {fp4_01, fp4_23}, {$r0};
                mov.b32 {sf_01, sf_23}, {$r1};
                cvt.rn.f16x2.e2m1x2 h_01, fp4_01;
                cvt.rn.f16x2.e2m1x2 h_23, fp4_23;
                cvt.rn.f16x2.e4m3x2 sf_h_01, sf_01;
                cvt.rn.f16x2.e4m3x2 sf_h_23, sf_23;
                mul.rn.f16x2 h_01, h_01, sf_h_01;
                mul.rn.f16x2 h_23, h_23, sf_h_23;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_01, h_01;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_23, h_23;
                mov.b32 {$w0}, {e4m3_01, e4m3_23};
            }
            """,
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )


class NVFP4DequantizeQmul4LinearKernel:
    """M-agnostic linear NVFP4 block dequantization kernel."""

    def __init__(self, K: int):
        if K <= 0 or K % _NVFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"K must be positive and divisible by {_NVFP4_BLOCK_SIZE}, got {K}"
            )
        self.K = K
        self.blocks_per_row = K // _NVFP4_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mScales: cute.Tensor,
        mOutput: cute.Tensor,
        total_sf_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(mInput, mScales, mOutput, total_sf_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=_BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mScales: cute.Tensor,
        mOutput: cute.Tensor,
        total_sf_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * _THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * _THREADS_PER_BLOCK
        while sf_idx < total_sf_blocks:
            row_idx = sf_idx // self.blocks_per_row
            block_idx = sf_idx % self.blocks_per_row

            # One thread owns one NVFP4 scale block: 16 logical values,
            # represented by 8 packed bytes and one E4M3 scale byte.
            row_input = mInput[row_idx, None]
            input_ptr = get_ptr_as_int64(row_input, block_idx * Int32(8))
            packed_0 = ld_global_nc_u32(input_ptr)
            packed_1 = ld_global_nc_u32(input_ptr + Int32(4))

            packed_sf = _broadcast_low_byte(Int32(mScales[row_idx, block_idx]))
            out_0 = _mul_packed_e2m1x4_e4m3x4(
                Int16(packed_0 & Uint32(0xFFFF)), packed_sf
            )
            out_1 = _mul_packed_e2m1x4_e4m3x4(
                Int16(packed_0 >> Int32(16)), packed_sf
            )
            out_2 = _mul_packed_e2m1x4_e4m3x4(
                Int16(packed_1 & Uint32(0xFFFF)), packed_sf
            )
            out_3 = _mul_packed_e2m1x4_e4m3x4(
                Int16(packed_1 >> Int32(16)), packed_sf
            )

            row_output = mOutput[row_idx, None]
            output_ptr = get_ptr_as_int64(
                row_output, block_idx * Int32(_NVFP4_BLOCK_SIZE)
            )
            st_global_v4_u32(
                output_ptr,
                Uint32(out_0),
                Uint32(out_1),
                Uint32(out_2),
                Uint32(out_3),
            )
            sf_idx = sf_idx + stride


class NVFP4ActivePageMaterializeQmul4Kernel:
    """Gather active NVFP4 pages and write TRTLLM FP8 K/V layout directly.

    One thread owns one 16-element scale block for both K and V.  The kernel
    reads the physical page id, gathers packed E2M1 data and E4M3 block scales,
    applies QMUL4, and writes the two final FP8 planes without intermediate
    gathered or dequantized tensors.
    """

    def __init__(self, K: int):
        if K <= 0 or K % (_NVFP4_BLOCK_SIZE * 4) != 0:
            raise ValueError(
                f"K must be positive and divisible by "
                f"{_NVFP4_BLOCK_SIZE * 4}, got {K}"
            )
        self.K = K
        self.blocks_per_row = K // _NVFP4_BLOCK_SIZE
        self.scale_quarter = self.blocks_per_row // 4

    @cute.jit
    def __call__(
        self,
        mKData: cute.Tensor,
        mVData: cute.Tensor,
        mKScales: cute.Tensor,
        mVScales: cute.Tensor,
        mPhysicalPageIds: cute.Tensor,
        mOutput: cute.Tensor,
        total_sf_blocks: Int32,
        physical_pages: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(
            mKData,
            mVData,
            mKScales,
            mVScales,
            mPhysicalPageIds,
            mOutput,
            total_sf_blocks,
            physical_pages,
        ).launch(
            grid=[num_blocks, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=_BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mKData: cute.Tensor,
        mVData: cute.Tensor,
        mKScales: cute.Tensor,
        mVScales: cute.Tensor,
        mPhysicalPageIds: cute.Tensor,
        mOutput: cute.Tensor,
        total_sf_blocks: Int32,
        physical_pages: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        num_heads = Int32(mKData.shape[1])
        page_size = Int32(mKData.shape[2])
        rows_per_page = num_heads * page_size
        sf_idx = bidx * _THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * _THREADS_PER_BLOCK
        while sf_idx < total_sf_blocks:
            row_idx = sf_idx // self.blocks_per_row
            block_idx = sf_idx % self.blocks_per_row
            compact_page = row_idx // rows_per_page
            row_in_page = row_idx % rows_per_page
            head_idx = row_in_page // page_size
            token_idx = row_in_page % page_size

            physical_page = Int32(mPhysicalPageIds[compact_page])
            physical_page = cutlass.max(
                Int32(0),
                cutlass.min(physical_page, physical_pages - Int32(1)),
            )

            k_input = mKData[physical_page, head_idx, token_idx, None]
            v_input = mVData[physical_page, head_idx, token_idx, None]
            input_byte = block_idx * Int32(8)
            k_input_ptr = get_ptr_as_int64(k_input, input_byte)
            v_input_ptr = get_ptr_as_int64(v_input, input_byte)
            k_packed_0 = ld_global_nc_u32(k_input_ptr)
            k_packed_1 = ld_global_nc_u32(k_input_ptr + Int32(4))
            v_packed_0 = ld_global_nc_u32(v_input_ptr)
            v_packed_1 = ld_global_nc_u32(v_input_ptr + Int32(4))

            k_sf = _broadcast_low_byte(
                Int32(mKScales[physical_page, head_idx, token_idx, block_idx])
            )
            # V block scales are stored in TRTLLM-GEN's 4-token-interleaved
            # HND layout.  Map the logical (token, scale-block) coordinate to
            # that physical coordinate instead of materializing an unswizzle.
            v_token_idx = (
                (token_idx // Int32(4)) * Int32(4)
                + block_idx // self.scale_quarter
            )
            v_scale_idx = (
                (block_idx % self.scale_quarter) * Int32(4)
                + token_idx % Int32(4)
            )
            v_sf = _broadcast_low_byte(
                Int32(mVScales[physical_page, head_idx, v_token_idx, v_scale_idx])
            )

            k_out_0 = _mul_packed_e2m1x4_e4m3x4(
                Int16(k_packed_0 & Uint32(0xFFFF)), k_sf
            )
            k_out_1 = _mul_packed_e2m1x4_e4m3x4(
                Int16(k_packed_0 >> Int32(16)), k_sf
            )
            k_out_2 = _mul_packed_e2m1x4_e4m3x4(
                Int16(k_packed_1 & Uint32(0xFFFF)), k_sf
            )
            k_out_3 = _mul_packed_e2m1x4_e4m3x4(
                Int16(k_packed_1 >> Int32(16)), k_sf
            )
            v_out_0 = _mul_packed_e2m1x4_e4m3x4(
                Int16(v_packed_0 & Uint32(0xFFFF)), v_sf
            )
            v_out_1 = _mul_packed_e2m1x4_e4m3x4(
                Int16(v_packed_0 >> Int32(16)), v_sf
            )
            v_out_2 = _mul_packed_e2m1x4_e4m3x4(
                Int16(v_packed_1 & Uint32(0xFFFF)), v_sf
            )
            v_out_3 = _mul_packed_e2m1x4_e4m3x4(
                Int16(v_packed_1 >> Int32(16)), v_sf
            )

            output_byte = block_idx * Int32(_NVFP4_BLOCK_SIZE)
            # Compact page zero is the TRTLLM sentinel; active pages start at
            # one and K/V occupy complete, adjacent planes.
            k_output = mOutput[
                compact_page + Int32(1), Int32(0), head_idx, token_idx, None
            ]
            v_output = mOutput[
                compact_page + Int32(1), Int32(1), head_idx, token_idx, None
            ]
            st_global_v4_u32(
                get_ptr_as_int64(k_output, output_byte),
                Uint32(k_out_0),
                Uint32(k_out_1),
                Uint32(k_out_2),
                Uint32(k_out_3),
            )
            st_global_v4_u32(
                get_ptr_as_int64(v_output, output_byte),
                Uint32(v_out_0),
                Uint32(v_out_1),
                Uint32(v_out_2),
                Uint32(v_out_3),
            )
            sf_idx = sf_idx + stride


@functools.cache
def _get_compiled_qmul4_dequant(K: int) -> tuple[Callable, int]:
    sym_m = cute.sym_int()
    input_fake = cute.runtime.make_fake_compact_tensor(
        Uint8, (sym_m, K // 2), stride_order=(1, 0), assumed_align=16
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        Uint8,
        (sym_m, K // _NVFP4_BLOCK_SIZE),
        stride_order=(1, 0),
        assumed_align=16,
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (sym_m, K),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel_obj = NVFP4DequantizeQmul4LinearKernel(K)
    compile_options = "--enable-tvm-ffi --opt-level 3"
    if os.getenv("CUTE_DSL_KEEP"):
        compile_options += " --keep-ptx --keep-cubin"
    def compile_kernel() -> Callable:
        return cute.compile(
            kernel_obj,
            input_fake,
            scales_fake,
            output_fake,
            Int32(1),
            Int32(1),
            stream_fake,
            options=compile_options,
        )

    compiled = build_and_load_cute_dsl_kernel(
        "nvfp4_qmul4_dequant",
        f"h{K}",
        compile_kernel,
        extra_key_files=_kernel_source_files(),
    )
    return compiled, _THREADS_PER_BLOCK


@functools.cache
def _get_compiled_active_page_materialize(K: int) -> tuple[Callable, int]:
    physical_pages = cute.sym_int()
    compact_pages = cute.sym_int()
    output_pages = cute.sym_int()
    num_heads = cute.sym_int()
    page_size = cute.sym_int(divisibility=4)
    k_data_outer_stride = cute.sym_int64(divisibility=1)
    v_data_outer_stride = cute.sym_int64(divisibility=1)
    k_sf_outer_stride = cute.sym_int64(divisibility=1)
    v_sf_outer_stride = cute.sym_int64(divisibility=1)
    data_dim = K // 2
    scale_dim = K // _NVFP4_BLOCK_SIZE

    def fake_paged(dtype, last_dim, outer_stride):
        return cute.runtime.make_fake_tensor(
            dtype,
            (physical_pages, num_heads, page_size, last_dim),
            stride=(
                outer_stride,
                page_size * last_dim,
                last_dim,
                1,
            ),
            assumed_align=16,
        )

    k_data_fake = fake_paged(Uint8, data_dim, k_data_outer_stride)
    v_data_fake = fake_paged(Uint8, data_dim, v_data_outer_stride)
    k_scales_fake = fake_paged(Uint8, scale_dim, k_sf_outer_stride)
    v_scales_fake = fake_paged(Uint8, scale_dim, v_sf_outer_stride)
    page_ids_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (compact_pages,), stride_order=(0,), assumed_align=4
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (output_pages, 2, num_heads, page_size, K),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel_obj = NVFP4ActivePageMaterializeQmul4Kernel(K)
    compile_options = "--enable-tvm-ffi --opt-level 3"
    if os.getenv("CUTE_DSL_KEEP"):
        compile_options += " --keep-ptx --keep-cubin"
    def compile_kernel() -> Callable:
        return cute.compile(
            kernel_obj,
            k_data_fake,
            v_data_fake,
            k_scales_fake,
            v_scales_fake,
            page_ids_fake,
            output_fake,
            Int32(1),
            Int32(1),
            Int32(1),
            stream_fake,
            options=compile_options,
        )

    compiled = build_and_load_cute_dsl_kernel(
        "nvfp4_active_page_materialize",
        f"h{K}",
        compile_kernel,
        extra_key_files=_kernel_source_files(),
    )
    return compiled, _THREADS_PER_BLOCK


@flashinfer_api
def nvfp4_kv_materialize_active_pages_qmul4(
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_block_scales: torch.Tensor,
    v_block_scales: torch.Tensor,
    physical_page_ids: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Gather active NVFP4 pages directly into TRTLLM FP8 K/V layout.

    The output has shape ``[num_compact_pages + 1, 2, H, page_size, K]``.
    Page zero is an uninitialized TRTLLM sentinel; active pages start at one.
    Only E2M1 values and their E4M3 block scales are multiplied.  Downstream
    FP8 FMHA must apply the original tensor-level K/V global scales.  A caller
    may supply an exact-shape contiguous ``output`` view to reuse workspace.
    """
    tensors = (k_data, v_data, k_block_scales, v_block_scales)
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("K/V data and block-scale tensors must all be 4D")
    if any(t.dtype != torch.uint8 for t in tensors):
        raise TypeError("K/V data and block-scale tensors must use torch.uint8")
    if physical_page_ids.ndim != 1 or physical_page_ids.dtype != torch.int32:
        raise TypeError("physical_page_ids must be a 1D torch.int32 tensor")
    if any(not t.is_cuda for t in (*tensors, physical_page_ids)):
        raise ValueError("all inputs must be CUDA tensors")
    device = k_data.device
    if any(t.device != device for t in (*tensors[1:], physical_page_ids)):
        raise ValueError("all inputs must be on the same CUDA device")
    if not physical_page_ids.is_contiguous():
        raise ValueError("physical_page_ids must be contiguous")

    if k_data.shape != v_data.shape:
        raise ValueError(
            f"K/V data shapes must match, got {k_data.shape} and {v_data.shape}"
        )
    if k_block_scales.shape != v_block_scales.shape:
        raise ValueError(
            "K/V block-scale shapes must match, got "
            f"{k_block_scales.shape} and {v_block_scales.shape}"
        )
    physical_pages, num_heads, page_size, packed_dim = k_data.shape
    K = packed_dim * 2
    expected_scale_shape = (
        physical_pages,
        num_heads,
        page_size,
        K // _NVFP4_BLOCK_SIZE,
    )
    if tuple(k_block_scales.shape) != expected_scale_shape:
        raise ValueError(
            f"block scales must have shape {expected_scale_shape}, got "
            f"{tuple(k_block_scales.shape)}"
        )
    if K <= 0 or K % (_NVFP4_BLOCK_SIZE * 4) != 0:
        raise ValueError(
            f"K must be positive and divisible by "
            f"{_NVFP4_BLOCK_SIZE * 4}, got {K}"
        )
    if page_size <= 0 or page_size % 4 != 0:
        raise ValueError(f"page_size must be positive and divisible by 4, got {page_size}")
    if physical_pages <= 0 or physical_page_ids.numel() <= 0:
        raise ValueError("physical and compact page counts must both be positive")

    data_inner_strides = (page_size * packed_dim, packed_dim, 1)
    scale_dim = K // _NVFP4_BLOCK_SIZE
    scale_inner_strides = (page_size * scale_dim, scale_dim, 1)
    for name, tensor, expected in (
        ("k_data", k_data, data_inner_strides),
        ("v_data", v_data, data_inner_strides),
        ("k_block_scales", k_block_scales, scale_inner_strides),
        ("v_block_scales", v_block_scales, scale_inner_strides),
    ):
        if tuple(tensor.stride()[1:]) != expected:
            raise ValueError(
                f"{name} inner strides must be {expected}, got {tensor.stride()[1:]}"
            )

    compact_pages = physical_page_ids.numel()
    expected_output_shape = (compact_pages + 1, 2, num_heads, page_size, K)
    if output is None:
        output = torch.empty(
            expected_output_shape,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
    else:
        if output.shape != expected_output_shape:
            raise ValueError(
                f"output must have shape {expected_output_shape}, got "
                f"{tuple(output.shape)}"
            )
        if output.dtype != torch.float8_e4m3fn:
            raise TypeError("output must use torch.float8_e4m3fn")
        if output.device != device or not output.is_cuda:
            raise ValueError("output must be on the same CUDA device as inputs")
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")
    total_sf_blocks = compact_pages * num_heads * page_size * scale_dim
    if total_sf_blocks >= 2**31:
        raise ValueError(
            f"materialization has too many scale blocks for Int32: {total_sf_blocks}"
        )
    kernel, threads_per_block = _get_compiled_active_page_materialize(K)
    target_grid = get_num_sm(device) * _BLOCKS_PER_SM
    num_blocks = min(
        (total_sf_blocks + threads_per_block - 1) // threads_per_block,
        target_grid,
    )
    kernel(
        k_data,
        v_data,
        k_block_scales,
        v_block_scales,
        physical_page_ids,
        output,
        total_sf_blocks,
        physical_pages,
        num_blocks,
    )
    return output


@flashinfer_api
def nvfp4_kv_dequantize_qmul4(
    fp4_data: torch.Tensor,
    block_scales: torch.Tensor,
) -> torch.Tensor:
    r"""Dequantize linear NVFP4 blocks directly to E4M3 with QMUL4.

    This function applies ``E2M1 value * E4M3 block_scale`` only.  It leaves
    the tensor-level/global scale for the downstream FP8 FMHA.
    """
    if fp4_data.ndim != 2 or block_scales.ndim != 2:
        raise ValueError("fp4_data and block_scales must both be 2D")
    if fp4_data.dtype != torch.uint8 or block_scales.dtype != torch.uint8:
        raise TypeError("fp4_data and block_scales must both use torch.uint8")
    if not fp4_data.is_cuda or not block_scales.is_cuda:
        raise ValueError("fp4_data and block_scales must both be CUDA tensors")
    if fp4_data.device != block_scales.device:
        raise ValueError("fp4_data and block_scales must be on the same device")
    if not fp4_data.is_contiguous() or not block_scales.is_contiguous():
        raise ValueError("fp4_data and block_scales must be contiguous")

    M = fp4_data.shape[0]
    K = fp4_data.shape[1] * 2
    if K <= 0 or K % _NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"K must be positive and divisible by {_NVFP4_BLOCK_SIZE}, got {K}"
        )
    expected_scales = (M, K // _NVFP4_BLOCK_SIZE)
    if tuple(block_scales.shape) != expected_scales:
        raise ValueError(
            f"block_scales must have shape {expected_scales}, got "
            f"{tuple(block_scales.shape)}"
        )

    output = torch.empty(
        (M, K), dtype=torch.float8_e4m3fn, device=fp4_data.device
    )
    total_sf_blocks = M * (K // _NVFP4_BLOCK_SIZE)
    kernel, threads_per_block = _get_compiled_qmul4_dequant(K)
    target_grid = get_num_sm(fp4_data.device) * _BLOCKS_PER_SM
    num_blocks = min(
        (total_sf_blocks + threads_per_block - 1) // threads_per_block,
        target_grid,
    )
    kernel(fp4_data, block_scales, output, total_sf_blocks, num_blocks)
    return output

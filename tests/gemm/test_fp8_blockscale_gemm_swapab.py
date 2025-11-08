"""
Test for FP8 block scale GEMM with automatic swapAB selection.

This test verifies that:
1. The kernel works correctly for small M (swapAB path)
2. The kernel works correctly for large M (normal path)
3. The output matches reference computation within acceptable tolerance
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import get_compute_capability


def quantize_fp8_blockwise(
    tensor: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 with per-block scaling.
    
    Args:
        tensor: Input tensor (BF16 or FP32)
        block_size: Size of blocks for scaling
        
    Returns:
        Tuple of (quantized_tensor, scales)
    """
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    
    # Reshape to blocks
    if len(orig_shape) == 2:
        M, K = orig_shape
        # Pad K to be divisible by block_size
        K_padded = ((K + block_size - 1) // block_size) * block_size
        if K_padded != K:
            tensor_padded = torch.zeros(M, K_padded, device=tensor.device, dtype=orig_dtype)
            tensor_padded[:, :K] = tensor
            tensor = tensor_padded
        
        # Reshape to (M, K//block_size, block_size)
        tensor_blocks = tensor.view(M, -1, block_size)
        
        # Compute per-block max absolute value
        block_absmax = tensor_blocks.abs().max(dim=-1, keepdim=True)[0]
        
        # Add small epsilon to avoid division by zero
        block_absmax = torch.clamp(block_absmax, min=1e-12)
        
        # FP8 E4M3 max value is 448
        scales = block_absmax / 448.0
        
        # Quantize
        tensor_scaled = tensor_blocks / scales
        tensor_fp8 = tensor_scaled.to(torch.float8_e4m3fn)
        
        # Reshape back
        tensor_fp8 = tensor_fp8.view(M, -1)[:, :K]
        scales = scales.squeeze(-1)  # (M, K//block_size)
        
        return tensor_fp8, scales
    else:
        raise ValueError(f"Unsupported shape: {orig_shape}")


def dequantize_fp8_blockwise(
    tensor_fp8: torch.Tensor, scales: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Dequantize FP8 tensor with per-block scaling."""
    M, K = tensor_fp8.shape
    K_blocks = scales.shape[1]
    
    # Reshape for broadcasting
    tensor_blocks = tensor_fp8.view(M, K_blocks, block_size)
    scales_expanded = scales.unsqueeze(-1)  # (M, K_blocks, 1)
    
    # Dequantize
    tensor_dequant = tensor_blocks.to(torch.float32) * scales_expanded
    return tensor_dequant.view(M, -1)[:, :K]


@pytest.mark.parametrize("m", [8, 16, 32, 64, 128])  # Test both swapAB (< 32) and normal (>= 32)
@pytest.mark.parametrize("n", [2048, 4096])
@pytest.mark.parametrize("k", [4096, 8192])
@pytest.mark.parametrize("use_bf16_inputs", [True, False])
def test_fp8_blockscale_gemm_swapab(
    m: int,
    n: int,
    k: int,
    use_bf16_inputs: bool,
):
    """Test FP8 blockscale GEMM with automatic swapAB selection."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 9 or compute_capability[1] != 0:
        pytest.skip("fp8_blockscale_gemm_swapab is only supported on SM90 (Hopper).")
    
    # Try to import - if it fails, the feature isn't built
    try:
        from flashinfer.gemm import fp8_blockscale_gemm_swapab
    except ImportError as e:
        pytest.skip(f"fp8_blockscale_gemm_swapab not available: {e}")
    
    torch.manual_seed(42)
    device = "cuda"
    
    # Create random tensors
    input_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    
    # Compute reference output
    # output = input @ weight.T
    reference = torch.mm(input_bf16, weight_bf16.transpose(-2, -1))
    
    if use_bf16_inputs:
        # Test with BF16 inputs (internal quantization)
        output = fp8_blockscale_gemm_swapab(
            input_bf16,
            weight_bf16,
            out_dtype=torch.bfloat16,
        )
    else:
        # Test with FP8 inputs (pre-quantized)
        block_size = 128
        input_fp8, input_scales = quantize_fp8_blockwise(input_bf16, block_size)
        weight_fp8, weight_scales = quantize_fp8_blockwise(weight_bf16, block_size)
        
        output = fp8_blockscale_gemm_swapab(
            input_fp8,
            weight_fp8,
            input_scale=input_scales,
            weight_scale=weight_scales,
            out_dtype=torch.bfloat16,
        )
    
    # Check output shape
    assert output.shape == (m, n), f"Expected shape {(m, n)}, got {output.shape}"
    
    # Check numerical accuracy
    cos_sim = F.cosine_similarity(reference.reshape(-1), output.reshape(-1), dim=0)
    
    # Print debug info for failed tests
    if cos_sim < 0.90:
        print(f"\nTest failed for m={m}, n={n}, k={k}, use_bf16={use_bf16_inputs}")
        print(f"Cosine similarity: {cos_sim.item():.4f}")
        print(f"Reference norm: {reference.norm().item():.4f}")
        print(f"Output norm: {output.norm().item():.4f}")
        print(f"Max diff: {(reference - output).abs().max().item():.4f}")
        print(f"SwapAB expected: {m < 32}")
    
    # FP8 quantization introduces some error, so we use a relaxed threshold
    # BF16 path should be more accurate
    threshold = 0.95 if use_bf16_inputs else 0.90
    assert cos_sim > threshold, (
        f"Cosine similarity {cos_sim.item():.4f} below threshold {threshold} "
        f"for m={m}, n={n}, k={k}, use_bf16={use_bf16_inputs}"
    )


@pytest.mark.parametrize("m,expected_swap", [(16, True), (32, False), (64, False)])
def test_swapab_threshold(m: int, expected_swap: bool):
    """
    Test that swapAB is correctly triggered based on M threshold.
    
    This is a basic smoke test - we can't easily verify which kernel path
    was taken without instrumenting the C++ code, but we can verify the
    output is correct.
    """
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 9 or compute_capability[1] != 0:
        pytest.skip("fp8_blockscale_gemm_swapab is only supported on SM90 (Hopper).")
    
    try:
        from flashinfer.gemm import fp8_blockscale_gemm_swapab
    except ImportError as e:
        pytest.skip(f"fp8_blockscale_gemm_swapab not available: {e}")
    
    torch.manual_seed(42)
    device = "cuda"
    n, k = 4096, 4096
    
    # Create random tensors
    input_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    
    # Run GEMM
    output = fp8_blockscale_gemm_swapab(input_bf16, weight_bf16)
    
    # Verify output
    reference = torch.mm(input_bf16, weight_bf16.transpose(-2, -1))
    cos_sim = F.cosine_similarity(reference.reshape(-1), output.reshape(-1), dim=0)
    
    print(f"\nM={m}, expected_swap={expected_swap}, cosine_sim={cos_sim.item():.4f}")
    assert cos_sim > 0.95, f"Output mismatch for M={m}"


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])


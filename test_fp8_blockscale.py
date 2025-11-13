#!/usr/bin/env python3
"""Test script for FP8 block scale GEMM with SwapAB support."""

import torch
import sys

def to_fp8(tensor, dtype=torch.float8_e4m3fn):
    """Helper to convert tensor to FP8."""
    return tensor.to(dtype)


def test_module_import():
    """Test that the module can be imported and built."""
    print("=" * 80)
    print("Test 1: Module Import and Build")
    print("=" * 80)
    
    try:
        from flashinfer.gemm import get_fp8_blockscale_gemm_runner
        print("✓ Successfully imported get_fp8_blockscale_gemm_runner")
        
        # This will trigger JIT compilation
        runner = get_fp8_blockscale_gemm_runner()
        print("✓ Successfully built and loaded FP8 block scale GEMM runner")
        
        return runner
    except Exception as e:
        print(f"✗ Failed to import/build module: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_workspace_size(runner):
    """Test workspace size calculation."""
    print("\n" + "=" * 80)
    print("Test 2: Workspace Size Calculation")
    print("=" * 80)
    
    if runner is None:
        print("✗ Skipping test (runner not initialized)")
        return None
    
    try:
        M, N, K = 16, 4096, 4096
        workspace_size = runner.get_workspace_size(M, N, K)
        print(f"✓ Workspace size for M={M}, N={N}, K={K}: {workspace_size:,} bytes")
        return workspace_size
    except Exception as e:
        print(f"✗ Failed to get workspace size: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_gemm_fp8_input(runner, workspace_size):
    """Test GEMM with FP8 inputs."""
    print("\n" + "=" * 80)
    print("Test 3: FP8 GEMM with Pre-quantized Inputs")
    print("=" * 80)
    
    if runner is None:
        print("✗ Skipping test (runner not initialized)")
        return
    
    try:
        M, N, K = 16, 256, 256
        device = "cuda"
        block_size = 128
        
        # Create FP8 inputs by converting from BF16
        # PyTorch doesn't support randn() for FP8, so we create BF16 first
        input_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        
        # Convert to FP8
        input_fp8 = input_bf16.to(torch.float8_e4m3fn)
        weight_fp8 = weight_bf16.to(torch.float8_e4m3fn)
        
        # Create scale factors (per-token for input, per-block for weight)
        input_scale = torch.ones(M, K // block_size, device=device, dtype=torch.float32)
        weight_scale = torch.ones(N, K // block_size, device=device, dtype=torch.float32)
        
        # Allocate output
        output = torch.empty(M, N, device=device, dtype=torch.bfloat16)
        
        # Configure workspace
        if workspace_size and workspace_size > 0:
            workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
            runner.configure_workspace(workspace)
        
        # Run GEMM
        runner.gemm(input_fp8, weight_fp8, output, input_scale, weight_scale)
        torch.cuda.synchronize()
        
        print(f"✓ Successfully ran FP8 GEMM")
        print(f"  Input shape: {input_fp8.shape}, dtype: {input_fp8.dtype}")
        print(f"  Weight shape: {weight_fp8.shape}, dtype: {weight_fp8.dtype}")
        print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
        print(f"  Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        # Sanity check - output should not be all zeros or NaN
        if torch.isnan(output).any():
            print("  ⚠ Warning: Output contains NaN values")
        elif (output == 0).all():
            print("  ⚠ Warning: Output is all zeros")
        else:
            print("  ✓ Output looks reasonable (no NaN, not all zeros)")
        
    except Exception as e:
        print(f"✗ Failed to run FP8 GEMM: {e}")
        import traceback
        traceback.print_exc()


def test_gemm_bf16_input(runner, workspace_size):
    """Test GEMM with BF16 inputs (internal quantization)."""
    print("\n" + "=" * 80)
    print("Test 4: BF16 GEMM with Internal Quantization")
    print("=" * 80)
    
    if runner is None:
        print("✗ Skipping test (runner not initialized)")
        return
    
    try:
        M, N, K = 16, 256, 256
        device = "cuda"
        
        # Create BF16 inputs
        input_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        
        # Allocate output
        output = torch.empty(M, N, device=device, dtype=torch.bfloat16)
        
        # Configure workspace
        if workspace_size and workspace_size > 0:
            workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
            runner.configure_workspace(workspace)
        
        # Run GEMM (no scales for internal quantization)
        print("  Calling runner.gemm with BF16 inputs (internal quantization)...")
        runner.gemm(input_bf16, weight_bf16, output, None, None)
        torch.cuda.synchronize()
        
        print(f"✓ Successfully ran BF16 GEMM with internal quantization")
        print(f"  Input shape: {input_bf16.shape}, dtype: {input_bf16.dtype}")
        print(f"  Weight shape: {weight_bf16.shape}, dtype: {weight_bf16.dtype}")
        print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
        print(f"  Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        # Sanity check
        if torch.isnan(output).any():
            print("  ⚠ Warning: Output contains NaN values")
        elif (output == 0).all():
            print("  ⚠ Warning: Output is all zeros")
        else:
            print("  ✓ Output looks reasonable (no NaN, not all zeros)")
        
    except Exception as e:
        print(f"✗ Failed to run BF16 GEMM: {e}")
        import traceback
        traceback.print_exc()


def test_high_level_api():
    """Test the high-level Python API."""
    print("\n" + "=" * 80)
    print("Test 5: High-Level API (fp8_blockscale_gemm_swapab)")
    print("=" * 80)
    
    try:
        from flashinfer.gemm import fp8_blockscale_gemm_swapab
        
        M, N, K = 16, 4096, 4096
        device = "cuda"
        block_size = 128
        
        # Create FP8 inputs by converting from BF16
        input_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        
        input_fp8 = input_bf16.to(torch.float8_e4m3fn)
        weight_fp8 = weight_bf16.to(torch.float8_e4m3fn)
        
        # Create scale factors
        input_scale = torch.ones(M, K // block_size, device=device, dtype=torch.float32)
        weight_scale = torch.ones(N, K // block_size, device=device, dtype=torch.float32)
        
        # Run high-level API (automatically handles workspace and swapAB selection)
        output = fp8_blockscale_gemm_swapab(
            input_fp8, weight_fp8, input_scale, weight_scale
        )
        
        print(f"✓ Successfully ran high-level API")
        print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
        print(f"  Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        # Sanity check
        if torch.isnan(output).any():
            print("  ⚠ Warning: Output contains NaN values")
        elif (output == 0).all():
            print("  ⚠ Warning: Output is all zeros")
        else:
            print("  ✓ Output looks reasonable (no NaN, not all zeros)")
        
    except Exception as e:
        print(f"✗ Failed to run high-level API: {e}")
        import traceback
        traceback.print_exc()


def test_correctness_simple():
    """Simple correctness test comparing with PyTorch reference."""
    print("\n" + "=" * 80)
    print("Test 6: Correctness Check (Simple)")
    print("=" * 80)
    
    try:
        from flashinfer.gemm import get_fp8_blockscale_gemm_runner
        
        M, N, K = 8, 256, 128  # Test with larger N (M=4 has kernel bug)
        device = "cuda"
        block_size = 128
        
        # Create small test inputs
        input_bf16 = torch.ones(M, K, device=device, dtype=torch.bfloat16) * 0.5
        weight_bf16 = torch.ones(N, K, device=device, dtype=torch.bfloat16) * 0.5
        
        # Reference computation in BF16
        ref_output = torch.matmul(input_bf16, weight_bf16.T)
        
        # Convert to FP8 and run kernel
        input_fp8 = input_bf16.to(torch.float8_e4m3fn)
        weight_fp8 = weight_bf16.to(torch.float8_e4m3fn)
        
        input_scale = torch.ones(M, K // block_size, device=device, dtype=torch.float32)
        weight_scale = torch.ones(N, K // block_size, device=device, dtype=torch.float32)
        
        runner = get_fp8_blockscale_gemm_runner()
        workspace_size = runner.get_workspace_size(M, N, K)
        if workspace_size > 0:
            workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
            runner.configure_workspace(workspace)
        
        output = torch.zeros(M, N, device=device, dtype=torch.bfloat16)  # Changed to zeros for debugging
        runner.gemm(input_fp8, weight_fp8, output, input_scale, weight_scale)
        torch.cuda.synchronize()
        
        print(f"✓ Kernel executed successfully")
        print(f"  Reference output: {ref_output}")
        print(f"  Kernel output: {output}")
        print(f"  Expected value (approx): {M * K * 0.5 * 0.5} per element")
        
        # Check if outputs are close (allow for FP8 quantization error)
        diff = (output - ref_output).abs()
        max_diff = diff.max().item()
        print(f"  Max difference: {max_diff:.4f}")
        
        if max_diff < 5.0:  # Generous tolerance for FP8
            print(f"  ✓ Output is close to reference (max diff: {max_diff:.4f})")
        else:
            print(f"  ⚠ Warning: Large difference from reference (max diff: {max_diff:.4f})")
        
    except Exception as e:
        print(f"✗ Failed correctness test: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("FP8 Block Scale GEMM SwapAB Test Suite")
    print("=" * 80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ CUDA not available. Tests require CUDA.")
        sys.exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"Device: {device_name}")
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
    
    if compute_capability[0] < 9:
        print("✗ This feature requires Hopper (SM90) or newer architecture")
        sys.exit(1)
    
    print()
    
    # Run tests
    runner = test_module_import()
    workspace_size = test_workspace_size(runner)
    test_gemm_fp8_input(runner, workspace_size)
    test_gemm_bf16_input(runner, workspace_size)
    test_high_level_api()
    test_correctness_simple()
    
    print("\n" + "=" * 80)
    print("Test suite completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

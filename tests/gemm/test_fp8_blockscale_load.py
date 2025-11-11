"""
Quick test to see if the FP8 block scale GEMM module can compile and load.
"""

import torch
from flashinfer.utils import get_compute_capability

device = torch.device("cuda")
compute_capability = get_compute_capability(device)
print(f"Compute capability: {compute_capability}")

if compute_capability[0] != 9 or compute_capability[1] != 0:
    print("This test requires SM90 (Hopper). Skipping.")
    exit(0)

print("Attempting to load FP8 block scale GEMM module...")
try:
    from flashinfer.jit.gemm.fp8_blockscale import gen_fp8_blockscale_gemm_sm90_module
    
    print("Generating JIT spec...")
    jit_spec = gen_fp8_blockscale_gemm_sm90_module()
    
    print("Building module...")
    module = jit_spec.build_and_load()
    
    print("✓ Module loaded successfully!")
    print(f"Module type: {type(module)}")
    
    print("\nInitializing runner...")
    runner = module.init()
    print(f"✓ Runner initialized: {type(runner)}")
    
    print("\n✓ All checks passed!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


#include <tvm/ffi/extra/module.h>
#include "../tvm_ffi_utils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"

namespace kernels = tensorrt_llm::kernels::fp8_blockscale_gemm;

using tvm::ffi::DLDataTypeToString;
using tvm::ffi::Function;
using tvm::ffi::Optional;

class Fp8BlockScaleGemmRunner : public tvm::ffi::ModuleObj {
 public:
  Fp8BlockScaleGemmRunner() {
    // Create the runner - we use bfloat16 activation, fp8 weight, bfloat16 output
    runner_ = std::make_unique<kernels::CutlassFp8BlockScaleGemmRunner<
        __nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>();
  }

  ~Fp8BlockScaleGemmRunner() override = default;

  const char* type_key() const final {
    return "flashinfer.Fp8BlockScaleGemmRunner";
  }

  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "gemm") {
      return Function::FromTyped(
          [this](TensorView input, TensorView weight, TensorView output,
                 Optional<TensorView> scales_a, Optional<TensorView> scales_b) {
            runGemm(input, weight, output, scales_a, scales_b);
          });
    } else if (name == "get_workspace_size") {
      return Function::FromTyped(
          [this](int64_t shape_m, int64_t shape_n, int64_t shape_k) -> int64_t {
            return getWorkspaceSize(shape_m, shape_n, shape_k);
          });
    } else if (name == "configure_workspace") {
      return Function::FromTyped(
          [this](TensorView workspace) {
            configureWorkspace(workspace);
          });
    }
    return NullOpt;
  }

 private:
  void runGemm(TensorView input, TensorView weight, TensorView output,
               Optional<TensorView> scales_a, Optional<TensorView> scales_b) {
    // Get CUDA stream from TVM runtime
    auto stream = reinterpret_cast<cudaStream_t>(tvm::ffi::get_cuda_stream());

    // Extract tensor info
    auto input_ptr = input.data_ptr();
    auto weight_ptr = weight.data_ptr();
    auto output_ptr = output.data_ptr();

    // Get dimensions: input is (M, K), weight is (N, K), output is (M, N)
    TVM_FFI_ICHECK(input.ndim() == 2) << "Input must be 2D (M, K)";
    TVM_FFI_ICHECK(weight.ndim() == 2) << "Weight must be 2D (N, K)";
    TVM_FFI_ICHECK(output.ndim() == 2) << "Output must be 2D (M, N)";

    int shape_m = input.shape(0);
    int shape_k = input.shape(1);
    int shape_n = weight.shape(0);

    TVM_FFI_ICHECK(weight.shape(1) == shape_k) << "Weight K dimension must match input K";
    TVM_FFI_ICHECK(output.shape(0) == shape_m) << "Output M dimension must match input M";
    TVM_FFI_ICHECK(output.shape(1) == shape_n) << "Output N dimension must match weight N";

    // Get scales if provided
    float const* scales_a_ptr = nullptr;
    float const* scales_b_ptr = nullptr;
    
    if (scales_a.defined()) {
      scales_a_ptr = reinterpret_cast<float const*>(scales_a.value().data_ptr());
    }
    
    if (scales_b.defined()) {
      scales_b_ptr = reinterpret_cast<float const*>(scales_b.value().data_ptr());
    }

    // Check input types
    if (input.dtype().code == kDLFloat && input.dtype().bits == 8) {
      // Input is FP8
      auto input_fp8 = reinterpret_cast<__nv_fp8_e4m3 const*>(input_ptr);
      auto weight_fp8 = reinterpret_cast<__nv_fp8_e4m3 const*>(weight_ptr);
      auto output_bf16 = reinterpret_cast<__nv_bfloat16*>(output_ptr);
      
      int ld_a = shape_k;
      int ld_b = shape_k;
      int ld_d = shape_n;
      
      runner_->gemm(input_fp8, ld_a, weight_fp8, ld_b, output_bf16, ld_d,
                    shape_m, shape_n, shape_k, scales_a_ptr, scales_b_ptr, stream);
    } else {
      // Input is BF16 or other, use the internal quantization path
      TVM_FFI_ICHECK(scales_a_ptr == nullptr && scales_b_ptr == nullptr) 
          << "Internal quantization path doesn't support external scales";
      
      runner_->gemm(output_ptr, input_ptr, weight_ptr, shape_m, shape_n, shape_k,
                    stream, scales_a_ptr, scales_b_ptr);
    }
  }

  int64_t getWorkspaceSize(int64_t shape_m, int64_t shape_n, int64_t shape_k) {
    return runner_->getWorkspaceSize(shape_m, shape_n, shape_k);
  }

  void configureWorkspace(TensorView workspace) {
    auto workspace_ptr = reinterpret_cast<char*>(workspace.data_ptr());
    runner_->configureWorkspace(workspace_ptr);
  }

  std::unique_ptr<kernels::CutlassFp8BlockScaleGemmRunnerInterface> runner_;
};

tvm::ffi::Module init() {
  auto ptr = tvm::ffi::make_object<Fp8BlockScaleGemmRunner>();
  return tvm::ffi::Module(ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);

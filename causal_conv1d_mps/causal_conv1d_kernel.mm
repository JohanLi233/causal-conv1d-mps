#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#include <string>
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <vector>
#include <cstring>
#include <unordered_map>

// Cache Metal device, queue, and pipelines to avoid repeated initialization overhead
static id<MTLDevice> g_device = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

// Convolution parameter struct
struct ConvParams {
    int batch_size;
    int dim;
    int seqlen;
    int width;
    bool silu_activation;
    
    // Strides
    int x_batch_stride;
    int x_c_stride;
    int x_l_stride;
    int weight_c_stride;
    int weight_width_stride;
    int out_batch_stride;
    int out_c_stride;
    int out_l_stride;
};

static void ensure_metal_pipeline_initialized(id<MTLDevice> device) {
  if (g_device != nil && !g_pipelines.empty()) {
    return;
  }

  g_device = device;
  TORCH_CHECK(g_device, "Failed to get default MTLDevice");

  // Load and compile Metal source
  std::ifstream file("causal_conv1d.metal");
  if (!file.good()) {
    const char *alt = std::getenv("CAUSAL_CONV1D_METAL_PATH");
    TORCH_CHECK(alt != nullptr,
                "Cannot open causal_conv1d.metal. Set env CAUSAL_CONV1D_METAL_PATH or run from the directory containing causal_conv1d.metal");
    file = std::ifstream(alt);
    TORCH_CHECK(file.good(), "Cannot open causal_conv1d.metal from CAUSAL_CONV1D_METAL_PATH");
  }
  std::string source((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
  NSString *librarySource = [NSString stringWithUTF8String:source.c_str()];

  NSError *error = nil;
  id<MTLLibrary> library = [g_device newLibraryWithSource:librarySource
                                                  options:nil
                                                    error:&error];
  TORCH_CHECK(library, "Failed to compile Metal library: ",
              [[error localizedDescription] UTF8String]);

  // Create compute pipelines
  std::vector<std::string> function_names = {
    "causal_conv1d_fwd_kernel",
    "causal_conv1d_simple_kernel",
    "causal_conv1d_simple_kernel_f16",
    "causal_conv1d_simple_kernel_bf16",
    "causal_conv1d_bwd_kernel",
    "causal_conv1d_bwd_kernel_f16",
    "causal_conv1d_bwd_kernel_bf16",
    // New optimized kernels
    "causal_conv1d_preact_grad_kernel",
    "causal_conv1d_preact_grad_kernel_f16", 
    "causal_conv1d_preact_grad_kernel_bf16",
    "causal_conv1d_bwd_kernel_hierarchical"
  };
  
  for (const auto& func_name : function_names) {
    NSString *functionName = [NSString stringWithUTF8String:func_name.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    TORCH_CHECK(function, "Failed to find Metal function '", func_name, "'");
    
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pipeline, "Failed to create compute pipeline state for '", func_name, "': ",
                [[error localizedDescription] UTF8String]);
    
    g_pipelines[func_name] = pipeline;
  }
}

// Setup convolution parameters
static ConvParams setup_conv_params(
    const torch::Tensor &x, 
    const torch::Tensor &weight,
    const torch::Tensor &out,
    bool silu_activation
) {
    ConvParams params;
    
    // Basic dimensions
    params.batch_size = x.size(0);
    params.dim = x.size(1);
    params.seqlen = x.size(2);
    params.width = weight.size(1);
    params.silu_activation = silu_activation;
    
    // Strides
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(2);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(2);
    
    return params;
}

// C++ wrapper for MPS causal conv1d kernel
torch::Tensor causal_conv1d_fwd_mps(
    const torch::Tensor &x, 
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    bool silu_activation = false
) {
  // Input validation
  TORCH_CHECK(x.device().is_mps(), "Tensor 'x' must be a MPS tensor");
  TORCH_CHECK(weight.device().is_mps(), "Tensor 'weight' must be a MPS tensor");
  TORCH_CHECK(x.is_contiguous(), "Tensor 'x' must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "Tensor 'weight' must be contiguous");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kHalf ||
          x.scalar_type() == torch::kBFloat16,
      "Only float32/float16/bfloat16 supported for x");
  TORCH_CHECK(
      weight.scalar_type() == x.scalar_type(),
      "weight dtype must match x dtype");
  
  // Shape checks
  TORCH_CHECK(x.dim() == 3, "Input tensor must have shape (batch, dim, seqlen)");
  TORCH_CHECK(weight.dim() == 2, "Weight tensor must have shape (dim, width)");
  TORCH_CHECK(x.size(1) == weight.size(0), "Input dim must match weight dim");
  
  if (bias.defined() && bias.numel() > 0) {
    TORCH_CHECK(bias.device().is_mps(), "Tensor 'bias' must be a MPS tensor");
    TORCH_CHECK(bias.is_contiguous(), "Tensor 'bias' must be contiguous");
    TORCH_CHECK(bias.scalar_type() == x.scalar_type(), "bias dtype must match x dtype");
    TORCH_CHECK(bias.dim() == 1, "Bias tensor must be 1D");
    TORCH_CHECK(bias.size(0) == x.size(1), "Bias dim must match input dim");
  }
  TORCH_CHECK(weight.size(1) == 4, "Simple kernel supports width=4");

  const int64_t batch_size = x.size(0);
  const int64_t dim = x.size(1);
  const int64_t seqlen = x.size(2);
  
  if (seqlen == 0) {
    return torch::empty_like(x);
  }

  // Use PyTorch current MPS stream
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  id<MTLDevice> device = (id<MTLDevice>)stream->device();
  ensure_metal_pipeline_initialized(device);

  // Create output tensor
  auto result_tensor = torch::empty_like(x);
  
  // Get Metal buffers
  id<MTLBuffer> x_buffer = (id<MTLBuffer>)x.storage().data();
  id<MTLBuffer> weight_buffer = (id<MTLBuffer>)weight.storage().data();
  id<MTLBuffer> bias_buffer = (bias.defined() && bias.numel() > 0) ? (id<MTLBuffer>)bias.storage().data() : nil;
  id<MTLBuffer> result_buffer = (id<MTLBuffer>)result_tensor.storage().data();
  
  // Compute byte offsets
  const NSUInteger x_offset = (NSUInteger)(x.storage_offset() * x.element_size());
  const NSUInteger weight_offset = (NSUInteger)(weight.storage_offset() * weight.element_size());
  const NSUInteger bias_offset = (bias.defined() && bias.numel() > 0) ? (NSUInteger)(bias.storage_offset() * bias.element_size()) : 0;
  const NSUInteger result_offset = (NSUInteger)(result_tensor.storage_offset() * result_tensor.element_size());

  // Note: removed unused params as we use a simplified kernel
  // ConvParams params = setup_conv_params(x, weight, result_tensor, silu_activation);

  // Select kernel by dtype
  id<MTLComputePipelineState> pipeline = nil;
  if (x.scalar_type() == torch::kFloat) {
    pipeline = g_pipelines["causal_conv1d_simple_kernel"];
  } else if (x.scalar_type() == torch::kHalf) {
    pipeline = g_pipelines["causal_conv1d_simple_kernel_f16"];
  } else if (x.scalar_type() == torch::kBFloat16) {
    pipeline = g_pipelines["causal_conv1d_simple_kernel_bf16"];
  } else {
    TORCH_CHECK(false, "Unsupported dtype");
  }
  
  // Encode compute command
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  [encoder setComputePipelineState:pipeline];
  
  // Set buffers
  [encoder setBuffer:x_buffer offset:x_offset atIndex:0];
  [encoder setBuffer:weight_buffer offset:weight_offset atIndex:1];
  if (bias_buffer != nil) {
    [encoder setBuffer:bias_buffer offset:bias_offset atIndex:2];
  } else {
    [encoder setBuffer:nil offset:0 atIndex:2];
  }
  [encoder setBuffer:result_buffer offset:result_offset atIndex:3];
  
  // Set parameters
  uint32_t batch_u32 = (uint32_t)batch_size;
  uint32_t dim_u32 = (uint32_t)dim;
  uint32_t seqlen_u32 = (uint32_t)seqlen;
  bool silu_flag = silu_activation;
  
  [encoder setBytes:&batch_u32 length:sizeof(uint32_t) atIndex:4];
  [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:5];
  [encoder setBytes:&seqlen_u32 length:sizeof(uint32_t) atIndex:6];
  [encoder setBytes:&silu_flag length:sizeof(bool) atIndex:7];

  // Compute threadgroup size
  // Use 3D grid: (batch_size, dim, seqlen)
  MTLSize gridSize = MTLSizeMake((NSUInteger)batch_size, (NSUInteger)dim, (NSUInteger)seqlen);
  
  // Choose threadgroup size (conservative)
  NSUInteger maxThreadsPerGroup = [pipeline maxTotalThreadsPerThreadgroup];
  NSUInteger threadsPerGroup = MIN(256, maxThreadsPerGroup);
  MTLSize threadgroupSize = MTLSizeMake(1, 1, threadsPerGroup);
  
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

  return result_tensor;
}

// Simplified interface function to match PyTorch autograd function expectations
torch::Tensor causal_conv1d_mps(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    const torch::Tensor &seq_idx,
    const torch::Tensor &initial_states,
    const torch::Tensor &final_states_out,
    bool silu_activation
) {
    // Only basic functionality is implemented; advanced parameters are ignored
    return causal_conv1d_fwd_mps(x, weight, bias, silu_activation);
}

// Helper to safely get MTLBuffer and offset
static inline std::pair<id<MTLBuffer>, NSUInteger> getBufferAndOffset(const torch::Tensor& t) {
    if (!t.defined() || t.numel() == 0) return {nil, 0};
    id<MTLBuffer> buffer = (id<MTLBuffer>)t.storage().data();
    NSUInteger offset = (NSUInteger)(t.storage_offset() * t.element_size());
    return {buffer, offset};
}

// =====================================================================================
// BACKWARD PASS IMPLEMENTATIONS
// =====================================================================================

// Optimized backward pass for basic causal conv1d using two-pass approach
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> causal_conv1d_bwd_mps(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    const torch::Tensor &grad_output,
    bool silu_activation
) {
    // Input validation
    TORCH_CHECK(x.device().is_mps(), "Tensor 'x' must be a MPS tensor");
    TORCH_CHECK(weight.device().is_mps(), "Tensor 'weight' must be a MPS tensor");
    TORCH_CHECK(grad_output.device().is_mps(), "Tensor 'grad_output' must be a MPS tensor");
    TORCH_CHECK(x.is_contiguous(), "Tensor 'x' must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Tensor 'weight' must be contiguous");
    TORCH_CHECK(grad_output.is_contiguous(), "Tensor 'grad_output' must be contiguous");
    
    const int64_t batch_size = x.size(0);
    const int64_t dim = x.size(1);
    const int64_t seqlen = x.size(2);
    const int64_t width = weight.size(1);
    
    TORCH_CHECK(width == 4, "Only width=4 supported in backward kernel");
    
    if (seqlen == 0) {
        auto dx = torch::empty_like(x);
        auto dweight = torch::zeros_like(weight);
        auto dbias = bias.defined() && bias.numel() > 0 ? torch::zeros_like(bias) : torch::tensor({});
        return std::make_tuple(dx, dweight, dbias);
    }

    // Use PyTorch's current MPS stream
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    id<MTLDevice> device = (id<MTLDevice>)stream->device();
    ensure_metal_pipeline_initialized(device);

    // Create output tensors
    auto dx = torch::empty_like(x);
    auto dweight = torch::zeros_like(weight, torch::dtype(torch::kFloat32));
    auto dbias = torch::Tensor();
    if (bias.defined() && bias.numel() > 0) {
        dbias = torch::zeros_like(bias, torch::dtype(torch::kFloat32));
    }

    // Two-pass optimization for SiLU backward pass:
    // Pass 1: Compute pre-activation gradients (O(W) complexity)
    // Pass 2: Use pre-computed gradients in backward pass (O(W) complexity)
    // Total: O(W) instead of O(W²)
    
    // Get Metal buffers and offsets
    auto [x_buf, x_off] = getBufferAndOffset(x);
    auto [w_buf, w_off] = getBufferAndOffset(weight);
    auto [grad_out_buf, grad_out_off] = getBufferAndOffset(grad_output);
    auto [dx_buf, dx_off] = getBufferAndOffset(dx);
    auto [dw_buf, dw_off] = getBufferAndOffset(dweight);
    
    // Forward bias buffer
    id<MTLBuffer> fwd_bias_buf = nil;
    NSUInteger fwd_bias_off = 0;
    if (bias.defined() && bias.numel() > 0) {
        auto [b_buf, b_off] = getBufferAndOffset(bias);
        fwd_bias_buf = b_buf;
        fwd_bias_off = b_off;
    }
    
    id<MTLBuffer> db_buf = nil;
    NSUInteger db_off = 0;
    if (dbias.defined()) {
        auto [bias_buf, bias_off] = getBufferAndOffset(dbias);
        db_buf = bias_buf;
        db_off = bias_off;
    }

    // Two-pass optimization: First pass computes pre-activation gradients
    auto d_preact = torch::empty_like(grad_output, torch::dtype(torch::kFloat32));
    
    // Set up parameters for both passes
    uint32_t batch_u32 = (uint32_t)batch_size;
    uint32_t dim_u32 = (uint32_t)dim;
    uint32_t seqlen_u32 = (uint32_t)seqlen;
    bool silu_flag = silu_activation;
    
    // Set up dispatch parameters
    MTLSize gridSize = MTLSizeMake((NSUInteger)batch_size, (NSUInteger)dim, (NSUInteger)seqlen);
    NSUInteger maxThreadsPerGroup = [g_pipelines["causal_conv1d_preact_grad_kernel"] maxTotalThreadsPerThreadgroup];
    NSUInteger threadsPerGroup = MIN(256, maxThreadsPerGroup);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, threadsPerGroup);
    
    // Pass 1: Compute pre-activation gradients using optimized kernel
    id<MTLComputePipelineState> preact_pipeline = nil;
    if (x.scalar_type() == torch::kFloat) {
        preact_pipeline = g_pipelines["causal_conv1d_preact_grad_kernel"];
    } else if (x.scalar_type() == torch::kHalf) {
        preact_pipeline = g_pipelines["causal_conv1d_preact_grad_kernel_f16"];
    } else if (x.scalar_type() == torch::kBFloat16) {
        preact_pipeline = g_pipelines["causal_conv1d_preact_grad_kernel_bf16"];
    } else {
        TORCH_CHECK(false, "Unsupported dtype for backward pass");
    }
    
    auto [d_preact_buf, d_preact_off] = getBufferAndOffset(d_preact);
    
    // Execute pre-activation gradient kernel
    id<MTLComputeCommandEncoder> preact_encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [preact_encoder setComputePipelineState:preact_pipeline];
    
    [preact_encoder setBuffer:x_buf offset:x_off atIndex:0];
    [preact_encoder setBuffer:w_buf offset:w_off atIndex:1];
    [preact_encoder setBuffer:fwd_bias_buf offset:fwd_bias_off atIndex:2];
    [preact_encoder setBuffer:grad_out_buf offset:grad_out_off atIndex:3];
    [preact_encoder setBuffer:d_preact_buf offset:d_preact_off atIndex:4];
    
    [preact_encoder setBytes:&batch_u32 length:sizeof(uint32_t) atIndex:5];
    [preact_encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:6];
    [preact_encoder setBytes:&seqlen_u32 length:sizeof(uint32_t) atIndex:7];
    [preact_encoder setBytes:&silu_flag length:sizeof(bool) atIndex:8];
    
    [preact_encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    // Pass 2: Use hierarchical backward kernel with pre-computed gradients
    id<MTLComputePipelineState> pipeline = g_pipelines["causal_conv1d_bwd_kernel_hierarchical"];

    // Execute optimized backward kernel using pre-computed gradients
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];

    // Set buffers for hierarchical backward kernel
    [encoder setBuffer:x_buf offset:x_off atIndex:0];
    [encoder setBuffer:w_buf offset:w_off atIndex:1];
    [encoder setBuffer:d_preact_buf offset:d_preact_off atIndex:2];  // Pre-computed activation gradients
    [encoder setBuffer:dx_buf offset:dx_off atIndex:3];
    [encoder setBuffer:dw_buf offset:dw_off atIndex:4];
    [encoder setBuffer:db_buf offset:db_off atIndex:5];

    // Set parameters for hierarchical backward kernel
    [encoder setBytes:&batch_u32 length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&seqlen_u32 length:sizeof(uint32_t) atIndex:8];

    // Use threadgroup-based dispatch for hierarchical kernel
    MTLSize hierarchical_gridSize = MTLSizeMake((NSUInteger)batch_size, (NSUInteger)dim, 1);
    NSUInteger hierarchical_maxThreadsPerGroup = [pipeline maxTotalThreadsPerThreadgroup];
    NSUInteger hierarchical_threadsPerGroup = MIN(256, hierarchical_maxThreadsPerGroup);
    MTLSize hierarchical_threadgroupSize = MTLSizeMake(hierarchical_threadsPerGroup, 1, 1);
    
    [encoder dispatchThreadgroups:hierarchical_gridSize threadsPerThreadgroup:hierarchical_threadgroupSize];

    // Convert accumulation dtypes back to match input dtypes
    dweight = dweight.to(weight.dtype());
    if (dbias.defined()) {
        dbias = dbias.to(bias.dtype());
    }

    return std::make_tuple(dx, dweight, dbias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Forward pass functions
  m.def("causal_conv1d_fwd", &causal_conv1d_fwd_mps, 
        "Causal Conv1D forward pass using Metal compute kernel (MPS)");
  m.def("causal_conv1d", &causal_conv1d_mps,
        "Causal Conv1D with full interface compatibility");
  
  // Backward pass functions
  m.def("causal_conv1d_bwd", &causal_conv1d_bwd_mps,
        "Causal Conv1D backward pass using Metal compute kernel (MPS)");
}

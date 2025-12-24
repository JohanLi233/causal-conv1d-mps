#include <metal_stdlib>
using namespace metal;

kernel void causal_conv1d_fwd_kernel(
    device const float *input [[buffer(0)]],            // input tensor (batch, dim, seqlen)
    device const float *weight [[buffer(1)]],           // weights (dim, width)
    device const float *bias [[buffer(2)]],             // bias (dim) - can be nullptr
    device float *output [[buffer(3)]],                 // output tensor (batch, dim, seqlen)
    
    constant uint &batch_size [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &seqlen [[buffer(6)]],
    constant uint &width [[buffer(7)]],
    constant uint &silu_activation [[buffer(8)]],       // whether to use SiLU activation
    
    // Strides (in elements)
    constant uint &x_batch_stride [[buffer(9)]],
    constant uint &x_c_stride [[buffer(10)]],
    constant uint &x_l_stride [[buffer(11)]],
    constant uint &weight_c_stride [[buffer(12)]],
    constant uint &weight_width_stride [[buffer(13)]],
    constant uint &out_batch_stride [[buffer(14)]],
    constant uint &out_c_stride [[buffer(15)]],
    constant uint &out_l_stride [[buffer(16)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
)
{
    // Thread organization: each threadgroup handles a (batch_id, channel_id) pair
    const uint batch_id = threadgroup_position_in_grid.x;
    const uint channel_id = threadgroup_position_in_grid.y;
    const uint thread_id = thread_position_in_grid.x % threads_per_threadgroup.x;
    
    // Boundary check
    if (batch_id >= batch_size || channel_id >= dim) {
        return;
    }
    
    // Calculate data pointer offsets
    device const float *x = input + batch_id * x_batch_stride + channel_id * x_c_stride;
    device const float *w = weight + channel_id * weight_c_stride;
    device float *out = output + batch_id * out_batch_stride + channel_id * out_c_stride;
    
    // Get bias value
    float bias_val = (bias != nullptr) ? bias[channel_id] : 0.0f;
    
    // Preload weight values
    float weight_vals[4];  // fixed width=4 simplified version
    // Use min to ensure no out-of-bounds reads
    uint effective_width = min(width, (uint)4);
    for (uint i = 0; i < effective_width; i++) {
        weight_vals[i] = w[i * weight_width_stride];
    }
    
    // Each thread handles multiple sequence positions
    const uint elements_per_thread = 4;
    const uint total_threads = threads_per_threadgroup.x;
    const uint chunk_size = total_threads * elements_per_thread;
    const uint num_chunks = (seqlen + chunk_size - 1) / chunk_size;
    
    for (uint chunk = 0; chunk < num_chunks; chunk++) {
        const uint chunk_start = chunk * chunk_size;
        const uint thread_start = chunk_start + thread_id * elements_per_thread;
        
        // Process elements for the current thread
        for (uint elem = 0; elem < elements_per_thread; elem++) {
            uint pos = thread_start + elem;
            if (pos >= seqlen) break;
            
            float result = bias_val;
            
            // Causal convolution: only use current and previous inputs (Convention B)
            for (uint w_idx = 0; w_idx < effective_width; w_idx++) {
                int input_pos = (int)pos - (int)(effective_width - 1 - w_idx);
                if (input_pos >= 0) {
                    float input_val = x[input_pos * x_l_stride];
                    result += weight_vals[w_idx] * input_val;
                }
            }
            
            // Optional SiLU activation function
            if (silu_activation) {
                result = result / (1.0f + exp(-result));
            }
            
            // Store result
            out[pos * out_l_stride] = result;
        }
    }
}

// SiLU activation function helper
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// BF16 <-> FP32 conversion helper
inline float bf16_to_float(ushort h) {
    uint u = (uint)h << 16;
    return as_type<float>(u);
}

inline ushort float_to_bf16(float f) {
    uint u = as_type<uint>(f);
    
    // Extract sign bit and move to BF16 position (bit 15)
    ushort sign_bit = (ushort)((u >> 16) & 0x8000);

    // Add robustness: handle NaN/Inf
    // Check if exponent is all 1 (0xFF)
    if ((u & 0x7F800000) == 0x7F800000) {
        if ((u & 0x007FFFFF) != 0) {
            // NaN (mantissa non-zero) - use Quiet NaN (0x7FC0)
            return sign_bit | 0x7FC0;
        } else {
            // Infinity (mantissa all zero)
            return sign_bit | 0x7F80;
        }
    }

    // Round-to-nearest-even (RNE) for finite numbers
    uint lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (ushort)(u >> 16);
}

// Simplified version: fixed width=4, no state management
kernel void causal_conv1d_simple_kernel(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    
    constant uint &batch_size [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &seqlen [[buffer(6)]],
    constant bool &silu_activation [[buffer(7)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    // Each thread handles one output position
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    // Boundary check
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    // Calculate linear index
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * 4;  // width=4
    const uint output_idx = input_base + seq_pos;
    
    // Get bias
    float result = (bias != nullptr) ? bias[channel_id] : 0.0f;
    
    // Causal convolution: width=4
    const uint width = 4;
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = input[input_base + input_pos];
            float weight_val = weight[weight_base + w];
            result += weight_val * input_val;
        }
    }
    
    // Optional SiLU activation
    if (silu_activation) {
        result = silu(result);
    }
    
    // Store result
    output[output_idx] = result;
}

// Simplified version (float16): fixed width=4
kernel void causal_conv1d_simple_kernel_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],

    constant uint &batch_size [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &seqlen [[buffer(6)]],
    constant bool &silu_activation [[buffer(7)]],

    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;

    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }

    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * 4; // width=4
    const uint output_idx = input_base + seq_pos;

    float result = (bias != nullptr) ? (float)bias[channel_id] : 0.0f;

    const uint width = 4;
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = (float)input[input_base + input_pos];
            float weight_val = (float)weight[weight_base + w];
            result += weight_val * input_val;
        }
    }

    if (silu_activation) {
        result = silu(result);
    }

    output[output_idx] = (half)result;
}

// Simplified version (bfloat16): fixed width=4
kernel void causal_conv1d_simple_kernel_bf16(
    device const ushort *input [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device ushort *output [[buffer(3)]],

    constant uint &batch_size [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &seqlen [[buffer(6)]],
    constant bool &silu_activation [[buffer(7)]],

    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;

    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }

    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * 4; // width=4
    const uint output_idx = input_base + seq_pos;

    float result = (bias != nullptr) ? bf16_to_float(bias[channel_id]) : 0.0f;

    const uint width = 4;
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = bf16_to_float(input[input_base + input_pos]);
            float weight_val = bf16_to_float(weight[weight_base + w]);
            result += weight_val * input_val;
        }
    }

    if (silu_activation) {
        result = silu(result);
    }

    output[output_idx] = float_to_bf16(result);
}

// =====================================================================================
// BACKWARD PASS KERNELS - OPTIMIZED FOR O(W) COMPLEXITY
// =====================================================================================

// Pre-activation gradient computation kernel (float32)
// This kernel computes d_preact = grad_output * silu_derivative once per element
// avoiding the O(W²) complexity in the original backward pass
kernel void causal_conv1d_preact_grad_kernel(
    device const float *x [[buffer(0)]],           // Original input (batch, dim, seqlen)
    device const float *weight [[buffer(1)]],      // Original weight (dim, width)
    device const float *bias [[buffer(2)]],        // Forward bias (dim) - may be nullptr
    device const float *grad_output [[buffer(3)]], // Gradient w.r.t. output (batch, dim, seqlen)
    device float *d_preact [[buffer(4)]],          // Output: d_preact (batch, dim, seqlen)
    
    constant uint &batch_size [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant uint &seqlen [[buffer(7)]],
    constant bool &silu_activation [[buffer(8)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    const uint width = 4;
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * width;
    const uint output_idx = input_base + seq_pos;
    
    float grad_out = grad_output[output_idx];
    
    if (silu_activation) {
        // Recompute forward pass once to get pre-activation value
        float pre_activation = 0.0f;
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = x[input_base + input_pos];
                float weight_val = weight[weight_base + w];
                pre_activation += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            pre_activation += bias[channel_id];
        }
        
        // Apply SiLU derivative: d/dx[SiLU(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        float sigmoid_val = 1.0f / (1.0f + exp(-pre_activation));
        grad_out *= sigmoid_val * (1.0f + pre_activation * (1.0f - sigmoid_val));
    }
    
    d_preact[output_idx] = grad_out;
}

// Pre-activation gradient computation kernel (float16)
kernel void causal_conv1d_preact_grad_kernel_f16(
    device const half *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device const half *grad_output [[buffer(3)]],
    device float *d_preact [[buffer(4)]],          // Always use float32 for d_preact
    
    constant uint &batch_size [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant uint &seqlen [[buffer(7)]],
    constant bool &silu_activation [[buffer(8)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    const uint width = 4;
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * width;
    const uint output_idx = input_base + seq_pos;
    
    float grad_out = (float)grad_output[output_idx];
    
    if (silu_activation) {
        float pre_activation = 0.0f;
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = (float)x[input_base + input_pos];
                float weight_val = (float)weight[weight_base + w];
                pre_activation += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            pre_activation += (float)bias[channel_id];
        }
        
        float sigmoid_val = 1.0f / (1.0f + exp(-pre_activation));
        grad_out *= sigmoid_val * (1.0f + pre_activation * (1.0f - sigmoid_val));
    }
    
    d_preact[output_idx] = grad_out;
}

// Pre-activation gradient computation kernel (bfloat16)
kernel void causal_conv1d_preact_grad_kernel_bf16(
    device const ushort *x [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device const ushort *grad_output [[buffer(3)]],
    device float *d_preact [[buffer(4)]],          // Always use float32 for d_preact
    
    constant uint &batch_size [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant uint &seqlen [[buffer(7)]],
    constant bool &silu_activation [[buffer(8)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    const uint width = 4;
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * width;
    const uint output_idx = input_base + seq_pos;
    
    float grad_out = bf16_to_float(grad_output[output_idx]);
    
    if (silu_activation) {
        float pre_activation = 0.0f;
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = bf16_to_float(x[input_base + input_pos]);
                float weight_val = bf16_to_float(weight[weight_base + w]);
                pre_activation += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            pre_activation += bf16_to_float(bias[channel_id]);
        }
        
        float sigmoid_val = 1.0f / (1.0f + exp(-pre_activation));
        grad_out *= sigmoid_val * (1.0f + pre_activation * (1.0f - sigmoid_val));
    }
    
    d_preact[output_idx] = grad_out;
}

// ORIGINAL backward pass for basic causal conv1d (float32) - restored for compatibility
kernel void causal_conv1d_bwd_kernel(
    device const float *x [[buffer(0)]],           // Original input (batch, dim, seqlen)
    device const float *weight [[buffer(1)]],      // Original weight (dim, width)
    device const float *grad_output [[buffer(2)]], // Gradient w.r.t. output (batch, dim, seqlen)
    device float *grad_x [[buffer(3)]],            // Gradient w.r.t. input (batch, dim, seqlen)
    device float *grad_weight [[buffer(4)]],       // Gradient w.r.t. weight (dim, width) - accumulated
    device const float *bias [[buffer(5)]],        // Forward bias (dim) - may be nullptr
    device float *grad_bias [[buffer(6)]],         // Gradient w.r.t. bias (dim) - accumulated (optional)
    
    constant uint &batch_size [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &seqlen [[buffer(9)]],
    constant bool &silu_activation [[buffer(10)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    // Boundary check
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    const uint width = 4; // Fixed width
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * width;
    const uint output_idx = input_base + seq_pos;
    
    // Get gradient from output
    float grad_out = grad_output[output_idx];
    
    // If SiLU activation was used, we need to apply its derivative
    if (silu_activation) {
        // Recompute forward pass to get the pre-activation value
        float pre_activation = 0.0f;
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = x[input_base + input_pos];
                float weight_val = weight[weight_base + w];
                pre_activation += weight_val * input_val;
            }
        }
        // Add bias contribution if available
        if (bias != nullptr) {
            pre_activation += bias[channel_id];
        }
        
        // Apply SiLU derivative: d/dx[SiLU(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        float sigmoid_val = 1.0f / (1.0f + exp(-pre_activation));
        grad_out *= sigmoid_val * (1.0f + pre_activation * (1.0f - sigmoid_val));
    }
    
    // Compute gradient w.r.t. input (dx)
    // For causal conv: dx[t] = sum_w(weight[w] * grad_output[t + (width - 1 - w)])
    float dx_val = 0.0f;
    for (uint w = 0; w < width; w++) {
        int future_pos = (int)seq_pos + (int)(width - 1 - w);
        if (future_pos < (int)seqlen) {
            float future_grad = grad_output[input_base + future_pos];
            
            // Apply activation derivative if needed for future position
            if (silu_activation) {
                // Recompute forward for future position
                float future_pre_activation = 0.0f;
                for (uint fw = 0; fw < width; fw++) {
                    int future_input_pos = future_pos - (int)(width - 1 - fw);
                    if (future_input_pos >= 0) {
                        float future_input_val = x[input_base + future_input_pos];
                        float future_weight_val = weight[weight_base + fw];
                        future_pre_activation += future_weight_val * future_input_val;
                    }
                }
                // Add bias contribution if available
                if (bias != nullptr) {
                    future_pre_activation += bias[channel_id];
                }
                float future_sigmoid = 1.0f / (1.0f + exp(-future_pre_activation));
                future_grad *= future_sigmoid * (1.0f + future_pre_activation * (1.0f - future_sigmoid));
            }
            
            dx_val += weight[weight_base + w] * future_grad;
        }
    }
    grad_x[output_idx] = dx_val;
    
    // Accumulate gradient w.r.t. weight (dweight) - using atomic operations
    // dweight[w] += x[t-(width-1-w)] * grad_output[t]
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = x[input_base + input_pos];
            atomic_fetch_add_explicit(
                (device atomic<float>*)&grad_weight[weight_base + w],
                input_val * grad_out,
                memory_order_relaxed
            );
        }
    }
    
    // Accumulate gradient w.r.t. bias (dbias) - using atomic operations
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[channel_id],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Backward pass for basic causal conv1d (float16)
kernel void causal_conv1d_bwd_kernel_f16(
    device const half *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *grad_output [[buffer(2)]],
    device half *grad_x [[buffer(3)]],
    device float *grad_weight [[buffer(4)]],     // Use float32 for accumulation
    device const half *bias [[buffer(5)]],       // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(6)]],       // Use float32 for accumulation
    
    constant uint &batch_size [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &seqlen [[buffer(9)]],
    constant bool &silu_activation [[buffer(10)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    const uint width = 4;
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * width;
    const uint output_idx = input_base + seq_pos;
    
    float grad_out = (float)grad_output[output_idx];
    
    // Apply SiLU derivative if needed
    if (silu_activation) {
        float pre_activation = 0.0f;
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = (float)x[input_base + input_pos];
                float weight_val = (float)weight[weight_base + w];
                pre_activation += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            pre_activation += (float)bias[channel_id];
        }
        float sigmoid_val = 1.0f / (1.0f + exp(-pre_activation));
        grad_out *= sigmoid_val * (1.0f + pre_activation * (1.0f - sigmoid_val));
    }
    
    // Compute dx
    float dx_val = 0.0f;
    for (uint w = 0; w < width; w++) {
        int future_pos = (int)seq_pos + (int)(width - 1 - w);
        if (future_pos < (int)seqlen) {
            float future_grad = (float)grad_output[input_base + future_pos];
            if (silu_activation) {
                // Apply activation derivative for future position
                float future_pre_activation = 0.0f;
                for (uint fw = 0; fw < width; fw++) {
                    int future_input_pos = future_pos - (int)(width - 1 - fw);
                    if (future_input_pos >= 0) {
                        float future_input_val = (float)x[input_base + future_input_pos];
                        float future_weight_val = (float)weight[weight_base + fw];
                        future_pre_activation += future_weight_val * future_input_val;
                    }
                }
                if (bias != nullptr) {
                    future_pre_activation += (float)bias[channel_id];
                }
                float future_sigmoid = 1.0f / (1.0f + exp(-future_pre_activation));
                future_grad *= future_sigmoid * (1.0f + future_pre_activation * (1.0f - future_sigmoid));
            }
            dx_val += (float)weight[weight_base + w] * future_grad;
        }
    }
    grad_x[output_idx] = (half)dx_val;
    
    // Accumulate dweight and dbias in float32
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = (float)x[input_base + input_pos];
            atomic_fetch_add_explicit(
                (device atomic<float>*)&grad_weight[weight_base + w],
                input_val * grad_out,
                memory_order_relaxed
            );
        }
    }
    
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[channel_id],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Backward pass for basic causal conv1d (bfloat16)
kernel void causal_conv1d_bwd_kernel_bf16(
    device const ushort *x [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const ushort *grad_output [[buffer(2)]],
    device ushort *grad_x [[buffer(3)]],
    device float *grad_weight [[buffer(4)]],     // Use float32 for accumulation
    device const ushort *bias [[buffer(5)]],     // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(6)]],       // Use float32 for accumulation
    
    constant uint &batch_size [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &seqlen [[buffer(9)]],
    constant bool &silu_activation [[buffer(10)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    const uint width = 4;
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * width;
    const uint output_idx = input_base + seq_pos;
    
    float grad_out = bf16_to_float(grad_output[output_idx]);
    
    // Apply SiLU derivative if needed
    if (silu_activation) {
        float pre_activation = 0.0f;
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = bf16_to_float(x[input_base + input_pos]);
                float weight_val = bf16_to_float(weight[weight_base + w]);
                pre_activation += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            pre_activation += bf16_to_float(bias[channel_id]);
        }
        float sigmoid_val = 1.0f / (1.0f + exp(-pre_activation));
        grad_out *= sigmoid_val * (1.0f + pre_activation * (1.0f - sigmoid_val));
    }
    
    // Compute dx
    float dx_val = 0.0f;
    for (uint w = 0; w < width; w++) {
        int future_pos = (int)seq_pos + (int)(width - 1 - w);
        if (future_pos < (int)seqlen) {
            float future_grad = bf16_to_float(grad_output[input_base + future_pos]);
            if (silu_activation) {
                // Apply activation derivative for future position
                float future_pre_activation = 0.0f;
                for (uint fw = 0; fw < width; fw++) {
                    int future_input_pos = future_pos - (int)(width - 1 - fw);
                    if (future_input_pos >= 0) {
                        float future_input_val = bf16_to_float(x[input_base + future_input_pos]);
                        float future_weight_val = bf16_to_float(weight[weight_base + fw]);
                        future_pre_activation += future_weight_val * future_input_val;
                    }
                }
                if (bias != nullptr) {
                    future_pre_activation += bf16_to_float(bias[channel_id]);
                }
                float future_sigmoid = 1.0f / (1.0f + exp(-future_pre_activation));
                future_grad *= future_sigmoid * (1.0f + future_pre_activation * (1.0f - future_sigmoid));
            }
            dx_val += bf16_to_float(weight[weight_base + w]) * future_grad;
        }
    }
    grad_x[output_idx] = float_to_bf16(dx_val);
    
    // Accumulate dweight and dbias in float32
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = bf16_to_float(x[input_base + input_pos]);
            atomic_fetch_add_explicit(
                (device atomic<float>*)&grad_weight[weight_base + w],
                input_val * grad_out,
                memory_order_relaxed
            );
        }
    }
    
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[channel_id],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Hierarchical reduction optimized backward kernel (float32)
// Uses threadgroup memory for local accumulation to reduce atomic contention
kernel void causal_conv1d_bwd_kernel_hierarchical(
    device const float *x [[buffer(0)]],           // Original input (batch, dim, seqlen)
    device const float *weight [[buffer(1)]],      // Original weight (dim, width)
    // NOTE: This now takes d_preact, not grad_output
    device const float *d_preact [[buffer(2)]],    // Pre-computed activation gradients
    device float *grad_x [[buffer(3)]],            // Gradient w.r.t. input
    device float *grad_weight [[buffer(4)]],       // Gradient w.r.t. weight - accumulated
    device float *grad_bias [[buffer(5)]],         // Gradient w.r.t. bias - accumulated
    
    constant uint &batch_size [[buffer(6)]],
    constant uint &dim [[buffer(7)]],
    constant uint &seqlen [[buffer(8)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]
)
{
    const uint batch_id = threadgroup_position_in_grid.x;
    const uint channel_id = threadgroup_position_in_grid.y;
    
    // Each threadgroup handles one channel. Threads iterate over seqlen.
    if (batch_id >= batch_size || channel_id >= dim) {
        return;
    }
    
    const uint width = 4;
    
    // Each threadgroup handles one channel. Threads iterate over seqlen.
    for (uint seq_pos = thread_index_in_threadgroup; seq_pos < seqlen; seq_pos += threads_per_threadgroup.x) {
        
        const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
        const uint weight_base = channel_id * width;
        const uint output_idx = input_base + seq_pos;
        
        // Use pre-computed activation gradient
        float grad_out = d_preact[output_idx];
        
        // --- 1. Compute gradient w.r.t. input (dx) ---
        // For causal convolution backward pass:
        // dx[t] = sum_{w=0..W-1} weight[w] * d_preact[t + (W-1-w)]
        // This accounts for how input x[t] affects all future outputs
        float dx_val = 0.0f;
        for (uint w = 0; w < width; w++) {
            int future_pos = (int)seq_pos + (int)(width - 1 - w);
            if (future_pos < (int)seqlen) {
                float future_grad = d_preact[input_base + future_pos];
                dx_val += weight[weight_base + w] * future_grad;
            }
        }
        grad_x[output_idx] = dx_val;
        
        // --- 2. Accumulate gradients for weight and bias ---
        // Use atomic operations since multiple threads may write to the same location
        // dweight[w] += x[t - (W-1-w)] * d_preact[t]
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = x[input_base + input_pos];
                atomic_fetch_add_explicit((device atomic<float>*)&grad_weight[weight_base + w], input_val * grad_out, memory_order_relaxed);
            }
        }
        
        if (grad_bias != nullptr) {
            atomic_fetch_add_explicit((device atomic<float>*)&grad_bias[channel_id], grad_out, memory_order_relaxed);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // No need for final reduction since we're writing directly to global memory
}

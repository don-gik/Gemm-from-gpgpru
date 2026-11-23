// --------------------------------
// Per workgroup (or tile)
static const uint BN = 64;
static const uint BM = 64;
static const uint BK = 16;
static const uint BN_VEC = BN / 4;
static const uint BM_VEC = BM / 4;
static const uint BK_VEC = BK / 4;
// Per thread
static const uint TM = 8;
static const uint TN = 4;
static const uint TN_VEC = TN / 4;
// Total thread count
static const uint NUM_THREADS = (BM * BN) / (TM * TN);

groupshared float4 As[BM][BK_VEC + 1];
groupshared float4 Bs[BK][BN_VEC + 1];

// Explicit descriptor bindings (Set 0)
[[vk::binding(0, 0)]] StructuredBuffer<float4> A;
[[vk::binding(1, 0)]] StructuredBuffer<float4> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> C;

// --------------------------------

[shader("compute")]
[numthreads(NUM_THREADS, 1, 1)]
void gemm(
    uint3 thread_idx: SV_GroupThreadID,
    uint3 block_idx: SV_GroupID,

    uniform uint M,
    uniform uint N,
    uniform uint K,
    uniform float alpha,
    uniform float beta,
) {
    let K_vec = K / 4;
    let N_vec = N / 4;
    
    // Setup some coordinates
    let c_row = block_idx.y;             // Tile coordinate from block
    let c_col = block_idx.x;
    let thread_row = thread_idx.x / (BN / TN); // Coalesced access of GMEM in workgroup
    let thread_col = thread_idx.x % (BN / TN);

    // Inner loading coordinates
    let inner_col_a = thread_idx.x % BK_VEC;
    let inner_row_a = thread_idx.x / BK_VEC;
    
    let inner_col_b = thread_idx.x % BN_VEC;
    let inner_row_b = thread_idx.x / BN_VEC;
    
    let jump_a = NUM_THREADS / BK_VEC;
    let jump_b = NUM_THREADS / BN_VEC;

    // Registers per thread
    float thread_results[TM * TN] = { 0.0 };

    for (uint k = 0; k < K; k += BK) {
        // Load tiles into groupshared memory.
        [unroll]
        for (uint off = 0; off < BM; off += jump_a) {
            let global_row_a = (c_row * BM) + inner_row_a + off;
            let global_col_a = (k / 4) + inner_col_a;
            As[inner_row_a + off][inner_col_a] = A[global_row_a * K_vec + global_col_a];
        }
        [unroll]
        for (uint off = 0; off < BK; off += jump_b) {
            let global_row_b = k + inner_row_b + off;
            let global_col_b = (c_col * BN_VEC) + inner_col_b;
            Bs[inner_row_b + off][inner_col_b] = B[global_row_b * N_vec + global_col_b];
        }

        GroupMemoryBarrierWithGroupSync();

        [loop]
        for (uint dv = 0; dv < BK_VEC; ++dv) {
            // As -> reg_m
            float4 reg_m[TM];
            [unroll]
            for (uint m = 0; m < TM; ++m) {
                reg_m[m] = As[thread_row * TM + m][dv];
            }

            // Transpose As in reg_m, use unroll for compile-time optimization
            // We intend to make this command into register swizzling
            [unroll]
            for (uint sub = 0; sub < 4; ++sub) {
                uint d = dv * 4 + sub;

                [unroll]
                for (uint nv = 0; nv < TN_VEC; ++nv) {
                    float4 reg_n = Bs[d][thread_col * TN_VEC + nv];

                    [unroll]
                    for (uint m = 0; m < TM; ++m) {
                        uint res_base = m * TN + nv * 4;
                        float a_val = reg_m[m][sub];

                        thread_results[res_base + 0] = fma(a_val, reg_n.x, thread_results[res_base + 0]);
                        thread_results[res_base + 1] = fma(a_val, reg_n.y, thread_results[res_base + 1]);
                        thread_results[res_base + 2] = fma(a_val, reg_n.z, thread_results[res_base + 2]);
                        thread_results[res_base + 3] = fma(a_val, reg_n.w, thread_results[res_base + 3]);
                    }
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write to C tile
    if (beta == 0.0) {
        [unroll]
        for (uint m = 0; m < TM; ++m) {
            let r = c_row * BM + thread_row * TM + m;
            [unroll]
            for (uint nv = 0; nv < TN_VEC; ++nv) {
                let cv = c_col * BN_VEC + thread_col * TN_VEC + nv;
                let res_idx = m * TN + nv * 4;
                float4 res_vec = float4(
                    thread_results[res_idx + 0],
                    thread_results[res_idx + 1],
                    thread_results[res_idx + 2],
                    thread_results[res_idx + 3]
                );
                C[r * N_vec + cv] = alpha * res_vec;
            }
        }
    } else {
        [unroll]
        for (uint m = 0; m < TM; ++m) {
            let r = c_row * BM + thread_row * TM + m;
            [unroll]
            for (uint nv = 0; nv < TN_VEC; ++nv) {
                let cv = c_col * BN_VEC + thread_col * TN_VEC + nv;
                let res_idx = m * TN + nv * 4;
                float4 res_vec = float4(
                    thread_results[res_idx + 0],
                    thread_results[res_idx + 1],
                    thread_results[res_idx + 2],
                    thread_results[res_idx + 3]
                );
                let c_idx = r * N_vec + cv;
                C[c_idx] = fma(float4(alpha), res_vec, beta * C[c_idx]);
            }
        }
    }
}

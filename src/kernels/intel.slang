// --------------------------------
// Per workgroup (or tile)
static const uint BN = 64;
static const uint BM = 64;
static const uint BK = 16;

// Per thread (TN = 4 -> 1 x float4 per row)
static const uint TM = 8;
static const uint TN = 4;

// Subgroup width
static const uint WAVE_SIZE = 8;

// Derived constants
static const uint NUM_THREADS = (BM * BN) / (TM * TN);
static const uint BK_VEC = BK / 4;
static const uint BN_VEC = BN / 4;
static const uint THREADS_PER_ROW = BN / TN;
static const uint GROUP_LANES = (THREADS_PER_ROW < WAVE_SIZE) ? THREADS_PER_ROW : WAVE_SIZE;
static const uint A_CHUNKS = (TM + GROUP_LANES - 1) / GROUP_LANES;

static const uint JUMP_A = NUM_THREADS / BK_VEC;
static const uint JUMP_B = NUM_THREADS / BN_VEC;
static const uint PREF_A = BM / JUMP_A;
static const uint PREF_B = BK / JUMP_B;

// L3 Cache Swizzling
static const uint GROUP_M = 8;

// +1 padding on As to break 64B (16-bank) stride and avoid 8-way SLM bank conflicts
groupshared float4 As[2][BM][BK_VEC + 1];
groupshared float4 Bs[2][BK][BN_VEC];

// --------------------------------

[[vk::binding(0, 0)]] StructuredBuffer<float4> A;
[[vk::binding(1, 0)]] StructuredBuffer<float4> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> C;

[shader("compute")]
[WaveSize(WAVE_SIZE)]
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
    let K_vec = K >> 2;
    let N_vec = N >> 2;

    // Swizzling for L3 cache - (num_tiles_n, num_tiles_m, 1)
    uint tile_m, tile_n4;
    {
        let num_tiles_m  = (M + BM - 1) / BM;
        let num_tiles_n  = (N + BN - 1) / BN;
        let linear_id    = block_idx.y * num_tiles_n + block_idx.x;
        let width        = num_tiles_n * GROUP_M;
        let group_id     = linear_id / width;
        let first_tile_m = group_id * GROUP_M;
        let group_size_m = min(num_tiles_m - first_tile_m, GROUP_M);

        tile_m  = (first_tile_m + (linear_id % group_size_m)) * BM;
        tile_n4 = (((linear_id % width) / group_size_m) * BN) >> 2;
    }

    // Thread & Wave coordinates
    let thread_row = thread_idx.x / THREADS_PER_ROW;
    let thread_col = thread_idx.x % THREADS_PER_ROW;

    let lane      = WaveGetLaneIndex();
    let lane_base = (lane / GROUP_LANES) * GROUP_LANES;
    let lane_off  = lane % GROUP_LANES;

    let inner_col_a = thread_idx.x % BK_VEC;
    let inner_row_a = thread_idx.x / BK_VEC;
    let inner_col_b = thread_idx.x % BN_VEC;
    let inner_row_b = thread_idx.x / BN_VEC;

    // Registers per thread: TM x float4
    float4 thread_results[TM] = {};

    // Prologue: load first K tile (k = 0)
    {
        [unroll]
        for (uint i = 0; i < PREF_A; ++i) {
            let r = i * JUMP_A + inner_row_a;
            As[0][r][inner_col_a] = A[(tile_m + r) * K_vec + inner_col_a];
        }
        [unroll]
        for (uint i = 0; i < PREF_B; ++i) {
            let r = i * JUMP_B + inner_row_b;
            Bs[0][r][inner_col_b] = B[r * N_vec + tile_n4 + inner_col_b];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint stage = 0;

    // Main software-pipelined loop (peeled last iteration to eliminate branches)
    for (uint k = 0; k + BK < K; k += BK) {
        float4 pref_a[PREF_A];
        float4 pref_b[PREF_B];

        let next_k     = k + BK;
        let next_k_vec = next_k >> 2;

        [unroll]
        for (uint i = 0; i < PREF_A; ++i) {
            let r = tile_m + i * JUMP_A + inner_row_a;
            pref_a[i] = A[r * K_vec + next_k_vec + inner_col_a];
        }
        [unroll]
        for (uint i = 0; i < PREF_B; ++i) {
            let r = next_k + i * JUMP_B + inner_row_b;
            pref_b[i] = B[r * N_vec + tile_n4 + inner_col_b];
        }

        [loop]
        for (uint dv = 0; dv < BK_VEC; ++dv) {
            float4 a_local[A_CHUNKS];

            [unroll]
            for (uint c = 0; c < A_CHUNKS; ++c) {
                uint m_idx = min(c * GROUP_LANES + lane_off, TM - 1);
                a_local[c] = As[stage][thread_row * TM + m_idx][dv];
            }

            [unroll]
            for (uint i = 0; i < 4; ++i) {
                float4 reg_b = Bs[stage][dv * 4 + i][thread_col];

                [unroll]
                for (uint m = 0; m < TM; ++m) {
                    let a_m = WaveReadLaneAt(
                        a_local[m / GROUP_LANES],
                        lane_base + (m % GROUP_LANES)
                    );
                    thread_results[m] = fma(float4(a_m[i]), reg_b, thread_results[m]);
                }
            }
        }

        uint next_stage = stage ^ 1;
        [unroll]
        for (uint i = 0; i < PREF_A; ++i) {
            As[next_stage][i * JUMP_A + inner_row_a][inner_col_a] = pref_a[i];
        }
        [unroll]
        for (uint i = 0; i < PREF_B; ++i) {
            Bs[next_stage][i * JUMP_B + inner_row_b][inner_col_b] = pref_b[i];
        }

        GroupMemoryBarrierWithGroupSync();
        stage = next_stage;
    }

    // Final K tile (no prefetch, no trailing barrier)
    [loop]
    for (uint dv = 0; dv < BK_VEC; ++dv) {
        float4 a_local[A_CHUNKS];

        [unroll]
        for (uint c = 0; c < A_CHUNKS; ++c) {
            uint m_idx = min(c * GROUP_LANES + lane_off, TM - 1);
            a_local[c] = As[stage][thread_row * TM + m_idx][dv];
        }

        [unroll]
        for (uint i = 0; i < 4; ++i) {
            float4 reg_b = Bs[stage][dv * 4 + i][thread_col];

            [unroll]
            for (uint m = 0; m < TM; ++m) {
                let a_m = WaveReadLaneAt(
                    a_local[m / GROUP_LANES],
                    lane_base + (m % GROUP_LANES)
                );
                thread_results[m] = fma(float4(a_m[i]), reg_b, thread_results[m]);
            }
        }
    }

    // Write to C tile
    let c_col4 = tile_n4 + thread_col;

    if (beta == 0.0) {
        [unroll]
        for (uint m = 0; m < TM; ++m) {
            let r = tile_m + thread_row * TM + m;
            C[r * N_vec + c_col4] = alpha * thread_results[m];
        }
    } else {
        [unroll]
        for (uint m = 0; m < TM; ++m) {
            let r = tile_m + thread_row * TM + m;
            let c_idx = r * N_vec + c_col4;
            C[c_idx] = fma(float4(alpha), thread_results[m], beta * C[c_idx]);
        }
    }
}

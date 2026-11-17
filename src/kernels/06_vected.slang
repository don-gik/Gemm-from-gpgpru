import lib.matrix;

// --------------------------------
// Per workgroup (or tile)
static const uint BN = 64;
static const uint BM = 64;
static const uint BK = 16;

// Per thread
static const uint TM = 8;
static const uint TN = 4;

// Subgroup width
static const uint WAVE_SIZE = 8;

// Derived constants
static const uint NUM_THREADS = (BM * BN) / (TM * TN);
static const uint BK_VEC = BK / 4;
static const uint BN_VEC = BN / 4;
static const uint TN_VEC = TN / 4;
static const uint THREADS_PER_ROW = BN / TN;
static const uint GROUP_LANES = (THREADS_PER_ROW < WAVE_SIZE) ? THREADS_PER_ROW : WAVE_SIZE;
static const uint A_CHUNKS = (TM + GROUP_LANES - 1) / GROUP_LANES;

static const uint JUMP_A = NUM_THREADS / BK_VEC;
static const uint JUMP_B = NUM_THREADS / BN_VEC;
static const uint PREF_A = BM / JUMP_A;
static const uint PREF_B = BK / JUMP_B;

// L3 Cache Swizzling
static const uint GROUP_M = 8;

groupshared float4 As[2][BM][BK_VEC];
groupshared float4 Bs[2][BK][BN_VEC];

// --------------------------------

[shader("compute")]
[WaveSize(WAVE_SIZE)]
[numthreads(NUM_THREADS, 1, 1)]
void gemm(
    uint3 thread_idx: SV_GroupThreadID,
    uint3 block_idx: SV_GroupID,

    uniform float4 *A,
    uniform float4 *B,
    uniform float4 *C,
    uniform uint M,
    uniform uint N,
    uniform uint K,
    uniform float alpha,
    uniform float beta,
) {
    // Wrap it into Mat structs for my fucking convenience - it is row major so let's be careful.
    let A_full = GlobalMat4(A, K >> 2);
    let B_full = GlobalMat4(B, N >> 2);
    let C_full = GlobalMat4(C, N >> 2);         // Now uses float4 aligned matrices

    // Swizzling for L3 cache       - (num_tiles_n, num_tiles_m, 1)
    uint tile_m, tile_n4;
    {
        let num_tiles_m  = (M + BM - 1) / BM;
        let num_tiles_n  = (N + BN - 1) / BN;
        let linear_id    = block_idx.y * num_tiles_n + block_idx.x;
        let width        = num_tiles_n * GROUP_M;
        let group_id     = linear_id / width;
        let first_tile_m = group_id * GROUP_M;
        let group_size_m = min(num_tiles_m - first_tile_m, GROUP_M);
    
        tile_m = (first_tile_m + (linear_id % group_size_m)) * BM;
        tile_n4 = (((linear_id % width) / group_size_m) * BN) >> 2;
    }

    // Now setup some coordinates
    let thread_row = thread_idx.x / (BN / TN);  // Coalesced access of GMEM in workgroup
    let thread_col = thread_idx.x % (BN / TN);
    
    let lane = WaveGetLaneIndex();
    let lane_base = (lane / GROUP_LANES) * GROUP_LANES;
    let lane_off  = lane % GROUP_LANES;

    let inner_col_a = thread_idx.x % BK_VEC;
    let inner_row_a = thread_idx.x / BK_VEC;
    let inner_col_b = thread_idx.x % BN_VEC;
    let inner_row_b = thread_idx.x / BN_VEC;

    let jump_a = NUM_THREADS / BK_VEC;
    let jump_b = NUM_THREADS / BN_VEC;

    // Registers per thread
    float4 thread_results[TM][TN_VEC] = {};

    let A_base = A_full.shift(tile_m, 0);
    let B_base = B_full.shift(0, tile_n4);

    // ----
    {
        [unroll]
        for (uint i = 0; i < PREF_A; ++i) {
            let r = i * JUMP_A + inner_row_a;
            As[0][r][inner_col_a] = A_base.load4(r, inner_col_a);
        }
        [unroll]
        for (uint i = 0; i < PREF_B; ++i) {
            let r = i * JUMP_B + inner_row_b;
            Bs[0][r][inner_col_b] = B_base.load4(r, inner_col_b);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint stage = 0;

    for (uint k = 0; k < K; k += BK) {
        float4 pref_a[PREF_A] = {};
        float4 pref_b[PREF_B] = {};

        let has_next = (k + BK < K);
        if (has_next) {
            let A_tile = A_base.shift(0, (k + BK) >> 2);
            let B_tile = B_base.shift(k + BK, 0);

            [unroll]
            for (uint i = 0; i < PREF_A; ++i) {
                pref_a[i] = A_tile.load4(i * JUMP_A + inner_row_a, inner_col_a);
            }
            [unroll]
            for (uint i = 0; i < PREF_B; ++i) {
                pref_b[i] = B_tile.load4(i * JUMP_B + inner_row_b, inner_col_b);
            }
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
                float4 reg_b[TN_VEC];
                [unroll]
                for (uint nv = 0; nv < TN_VEC; ++nv) {
                    reg_b[nv] = Bs[stage][dv * 4 + i][thread_col * TN_VEC + nv];
                }

                [unroll]
                for (uint m = 0; m < TM; ++m) {
                    let a_m = WaveReadLaneAt(
                        a_local[m / GROUP_LANES],
                        lane_base + (m % GROUP_LANES)
                    );

                    [unroll]
                    for (uint nv = 0; nv < TN_VEC; ++nv) {
                        thread_results[m][nv] = fma(float4(a_m[i]), reg_b[nv], thread_results[m][nv]);
                    }
                }
            }
        }

        if (has_next) {
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
    }

    // Get C tile view
    var C_tile = C_full.shift(tile_m, tile_n4);

    // Write to C tile
    [unroll]
    for (uint m = 0; m < TM; ++m) {
        let r = thread_row * TM + m;

        [unroll]
        for (uint nv = 0; nv < TN_VEC; ++nv) {
            let c4 = thread_col * TN_VEC + nv;

            if (beta == 0.0) {
                C_tile.store4(r, c4, alpha * thread_results[m][nv]);
            } else {
                float4 oldC = C_tile.load4(r, c4);
                C_tile.store4(r, c4, alpha * thread_results[m][nv] + beta * oldC);
            }
        }
    }
}

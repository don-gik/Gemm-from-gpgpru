// --------------------------------
// Per workgroup (or tile)
static const uint BN = 32;
static const uint BM = 32;
static const uint BK = 8;
// Per thread
static const uint TM = 4;
static const uint TN = 2;
// Total thread count
static const uint NUM_THREADS = (BM * BN) / (TM * TN);

groupshared float As[BM][BK];
groupshared float Bs[BK][BN];

// --------------------------------

[shader("compute")]
[numthreads(NUM_THREADS, 1, 1)]
void gemm(
    uint3 thread_idx: SV_GroupThreadID,
    uint3 block_idx: SV_GroupID,

    uniform float *A,
    uniform float *B,
    uniform float *C,
    uniform uint M,
    uniform uint N,
    uniform uint K,
    uniform float alpha,
    uniform float beta,
) {
    // Setup some coordinates
    let c_row = block_idx.y;             // Tile coordinate from block
    let c_col = block_idx.x;
    let thread_row = thread_idx.x / (BN / TN); // Coalesced access of GMEM in workgroup
    let thread_col = thread_idx.x % (BN / TN);

    // Inner loading coordinates
    let inner_col_a = thread_idx.x % BK;
    let inner_row_a = thread_idx.x / BK;
    
    let inner_col_b = thread_idx.x % BN;
    let inner_row_b = thread_idx.x / BN;
    
    let jump_a = NUM_THREADS / BK;
    let jump_b = NUM_THREADS / BN;

    // Registers per thread
    float thread_results[TM * TN] = { 0.0 };

    for (uint k = 0; k < K; k += BK) {
        // Get view for this tile - iterates over A & B.
        let tile_a = c_row * BM * M + k;
        let B_tile = c_col * BN * K + k;

        // Load tiles into groupshared memory.
        [unroll]
        for (uint off = 0; off < BM; off += jump_a) {
            let global_row_a = (c_row * BM) + inner_row_a + off;
            let global_col_a = k + inner_col_a;
            
            As[inner_row_a + off][inner_col_a] = 
                A[global_row_a * K + global_col_a];
        }
        [unroll]
        for (uint off = 0; off < BK; off += jump_b) {
            let global_row_b = k + inner_row_b + off;
            let global_col_b = (c_col * BN) + inner_col_b;

            Bs[inner_row_b + off][inner_col_b] = 
                B[global_row_b * N + global_col_b];
        }

        GroupMemoryBarrierWithGroupSync();

        [loop]
        for (uint d = 0; d < BK; ++d) {
            float reg_m[TM] = { 0.0 };
            float reg_n[TN] = { 0.0 };

            [unroll]
            for (uint i = 0; i < TM; ++i) {
                reg_m[i] = As[thread_row * TM + i][d];
            }
            [unroll]
            for (uint i = 0; i < TN; ++i) {
                reg_n[i] = Bs[d][thread_col * TN + i];
            }
            [unroll]
            for (uint m = 0; m < TM; ++m) {
                [unroll]
                for (uint n = 0; n < TN; ++n) {
                    thread_results[m * TN + n] += reg_m[m] * reg_n[n];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write to C tile
    [unroll]
    for (uint m = 0; m < TM; ++m) {
        [unroll]
        for (uint n = 0; n < TN; ++n) {
            let r = thread_row * TM + m + c_row * BM;
            let c = thread_col * TN + n + c_col * BN;
            C[(r * N) + c] = 
                alpha * thread_results[(m * TN) + n] 
                + beta * C[(r * N) + c];
        }
    }
}

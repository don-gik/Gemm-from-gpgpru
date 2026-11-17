// I'm fucking stressed reading spaghetti CUDA codes so for now I'LL USE STRUCTS

import lib.matrix;

// --------------------------------
// Per workgroup (or tile)
static const uint BN = 64;
static const uint BM = 64;
static const uint BK = 8;
// Per thread
static const uint TM = 4;
static const uint TN = 4;
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
    // Wrap it into Mat structs for my fucking convenience - it is row major so let's be careful.
    let A_full = GlobalMat(A, K);       // A: M x K
    let B_full = GlobalMat(B, N);       // B: K x N
    let C_full = GlobalMat(C, N);       // C: M x N

    // Now setup some coordinates
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

    // Get view for tile C
    var C_tile = C_full.shift(c_row * BM, c_col * BN);

    // Registers per thread
    float thread_results[TM * TN] = { 0.0 };
    float reg_m[TM], reg_n[TN];

    for (uint k = 0; k < K; k += BK) {
        // Get view for this tile - iterates over A & B.
        let A_tile = A_full.shift(c_row * BM, k);
        let B_tile = B_full.shift(k, c_col * BN);

        // Load tiles into groupshared memory.
        for (uint off = 0; off < BM; off += jump_a) {
            As[inner_row_a + off][inner_col_a] = A_tile[inner_row_a + off, inner_col_a];
        }
        for (uint off = 0; off < BK; off += jump_b) {
            Bs[inner_row_b + off][inner_col_b] = B_tile[inner_row_b + off, inner_col_b];
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint d = 0; d < BK; ++d) {
            for (uint i = 0; i < TM; ++i) {
                reg_m[i] = As[thread_row * TM + i][d];
            }
            for (uint i = 0; i < TN; ++i) {
                reg_n[i] = Bs[d][thread_col * TN + i];
            }
            for (uint m = 0; m < TM; ++m) {
                for (uint n = 0; n < TN; ++n) {
                    thread_results[m * TN + n] += reg_m[m] * reg_n[n];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write to C tile
    for (uint m = 0; m < TM; ++m) {
        for (uint n = 0; n < TN; ++n) {
            uint r = thread_row * TM + m;
            uint c = thread_col * TN + n;
            C_tile[r, c] = alpha * thread_results[m*TN+n] + beta * C_tile[r, c];
        }
    }
}

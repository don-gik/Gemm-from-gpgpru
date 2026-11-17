static const uint BN = 64;
static const uint BM = 64;
static const uint BK = 8;
static const uint TM = 8;

groupshared float As[BM * BK];
groupshared float Bs[BK * BN];

[shader("compute")]
[numthreads((BM * BN) / TM, 1, 1)]
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
    // Swapping Row & Col results in better sequential memory access.
    // Since memory is contiguous in row.
    const uint c_row = block_idx.y;
    const uint c_col = block_idx.x;

    // Coalesced access of GMEM - row is continuous
    const uint thread_row = thread_idx.x / BN;
    const uint thread_col = thread_idx.x % BN;

    // Advance pointers
    A = A + c_row * BM * K;                         // Row = c_row, Col = 0
    B = B + c_col * BN;                             // Row = 0,     Col = c_col
    C = C + c_row * BM * N + c_col * BN;            // Row = c_row, Col = c_col

    const uint inner_col_a = thread_idx.x % BK; // warp-level GMEM coalescing
    const uint inner_row_a = thread_idx.x / BK;
    const uint inner_col_b = thread_idx.x % BN; // warp-level GMEM coalescing
    const uint inner_row_b = thread_idx.x / BN;

    float thread_results[TM] = { 0.0 };     // Not tmp, now TM results per thread. (1D tiling inner loop)

    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK) {
        As[inner_row_a * BK + inner_col_a] = A[inner_row_a * K + inner_col_a];
        Bs[inner_row_b * BN + inner_col_b] = B[inner_row_b * N + inner_col_b];

        GroupMemoryBarrierWithGroupSync();

        A = A + BK;
        B = B + BK * N;

        // Execute dot product on current cached block
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            float tmp_b = Bs[dot_idx * BN + thread_col];

            for (uint res_idx = 0; res_idx < TM; ++res_idx) {
                thread_results[res_idx] += 
                    As[(thread_row * TM + res_idx) * BK + dot_idx] * tmp_b;
            }
        }
        
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint res_idx = 0; res_idx < TM; ++res_idx) {
        C[(thread_row * TM + res_idx) * N + thread_col] =
            alpha * thread_results[res_idx] +
            beta * C[(thread_row * TM + res_idx) * N + thread_col];
    }
}

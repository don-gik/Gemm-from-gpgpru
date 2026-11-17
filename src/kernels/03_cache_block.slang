static const uint BLOCK_SIZE = 32;

groupshared float As[BLOCK_SIZE * BLOCK_SIZE];
groupshared float Bs[BLOCK_SIZE * BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE * BLOCK_SIZE, 1, 1)]
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
    const uint c_row = block_idx.x;
    const uint c_col = block_idx.y;

    // Coalesced access of GMEM - row is continuous
    const uint thread_row = thread_idx.x / BLOCK_SIZE;
    const uint thread_col = thread_idx.x % BLOCK_SIZE;

    // Advance pointers
    A = A + c_row * BLOCK_SIZE * K;                         // Row = c_row, Col = 0
    B = B + c_col * BLOCK_SIZE;                             // Row = 0,     Col = c_col
    C = C + c_row * BLOCK_SIZE * N + c_col * BLOCK_SIZE;    // Row = c_row, Col = c_col

    float tmp = 0.0;
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BLOCK_SIZE) {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[thread_row * BLOCK_SIZE + thread_col] = A[thread_row * K + thread_col];
        Bs[thread_row * BLOCK_SIZE + thread_col] = B[thread_row * N + thread_col];

        GroupMemoryBarrierWithGroupSync();

        A = A + BLOCK_SIZE;
        B = B + BLOCK_SIZE * N;

        // Execute dot product on current cached block
        for (uint dot_idx = 0; dot_idx < BLOCK_SIZE; ++dot_idx) {
            tmp += As[thread_row * BLOCK_SIZE + dot_idx] *
                    Bs[dot_idx * BLOCK_SIZE + thread_col];
        }
        
        GroupMemoryBarrierWithGroupSync();
    }
    
    C[thread_row * N + thread_col] =
        alpha * tmp + beta * C[thread_row * N + thread_col];
}

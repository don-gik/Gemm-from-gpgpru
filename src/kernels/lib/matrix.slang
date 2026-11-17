struct GlobalMat {
    float* ptr;
    uint stride;

    __init(float* p, uint s) {
        ptr = p;
        stride = s;
    }

    __subscript(uint r, uint c) -> float {
        get { return ptr[r * stride + c]; }
        set { ptr[r * stride + c] = newValue; }
    }

    GlobalMat shift(uint r0, uint c0) {
        return { ptr + r0 * stride + c0, stride };
    }

    float4 load4(uint row, uint col) {
        return *reinterpret<float4*>(ptr + row * stride + col);
    }
    
    void store4(uint row, uint col, float4 val) {
        *reinterpret<float4*>(ptr + row * stride + col) = val;
    }
}

struct GlobalMat4 {
    float4* ptr;
    uint stride4;

    __init(float4* p, uint s4) {
        ptr = p;
        stride4 = s4;
    }

    GlobalMat4 shift(uint r0, uint c0_4) {
        return { ptr + r0 * stride4 + c0_4, stride4 };
    }

    float4 load4(uint row, uint col4) {
        return ptr[row * stride4 + col4];
    }

    void store4(uint row, uint col4, float4 val) {
        ptr[row * stride4 + col4] = val;
    }
}

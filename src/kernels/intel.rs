use super::*;

#[kernel("src/kernels/intel.slang", "gemm", profiles = [SpvGroupNonUniform, SpvGroupNonUniformShuffle])]
fn intel_slang(
    #[raw] A: &GpuSlice<f32>,
    #[raw] B: &GpuSlice<f32>,
    #[raw] C: &mut GpuSlice<f32>,
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    beta: f32,
) {
}

pub struct IntelKernel {}

impl IntelKernel {
    pub fn new() -> Self {
        IntelKernel {}
    }
}

impl GemmKernel for IntelKernel {
    fn name(&self) -> &str {
        "intel"
    }

    fn groups(&self, m: u32, n: u32, _k: u32) -> (u32, u32, u32) {
        (m.div_ceil(64), n.div_ceil(64), 1)
    }

    fn launch(
        &self,
        device: &Device,
        buf_a: &GpuSlice<f32>,
        buf_b: &GpuSlice<f32>,
        buf_c: &mut GpuSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let groups = self.groups(m, n, k);
        intel_slang(buf_a, buf_b, buf_c, m, n, k, alpha, beta).queue(device, groups)?;
        Ok(())
    }
}

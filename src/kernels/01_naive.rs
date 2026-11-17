use super::*;

#[kernel("src/kernels/01_naive.slang", "gemm")]
fn naive_slang(
    A: &GpuSlice<f32>,
    B: &GpuSlice<f32>,
    C: &mut GpuSlice<f32>,
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    beta: f32,
) {
}

pub struct NaiveKernel {}

impl NaiveKernel {
    pub fn new() -> Self {
        NaiveKernel {}
    }
}

impl GemmKernel for NaiveKernel {
    fn name(&self) -> &str {
        "naive"
    }

    fn groups(&self, m: u32, n: u32, _k: u32) -> (u32, u32, u32) {
        (m.div_ceil(32), n.div_ceil(32), 1)
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
        naive_slang(buf_a, buf_b, buf_c, m, n, k, alpha, beta).queue(device, groups)?;
        Ok(())
    }
}

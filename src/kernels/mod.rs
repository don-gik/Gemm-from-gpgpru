#[path = "03_cache_block.rs"]
mod cache_block;
#[path = "02_coalesced.rs"]
mod coalesced;
#[path = "01_naive.rs"]
mod naive;
#[path = "04_tile1d.rs"]
mod tile1d;
#[path = "05_tile2d.rs"]
mod tile2d;
#[path = "06_vected.rs"]
mod vected;

pub use cache_block::*;
pub use coalesced::*;
pub use naive::*;
pub use tile1d::*;
pub use tile2d::*;
pub use vected::*;

use super::*;

pub trait GemmKernel {
    fn name(&self) -> &str;
    fn groups(&self, m: u32, n: u32, k: u32) -> (u32, u32, u32);
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
    ) -> Result<()>;
}

// ----------------------------------------------------------------

pub struct GemmBenchmark<'a> {
    device: &'a Device,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub alpha: f32,
    pub beta: f32,
}

impl<'a> GemmBenchmark<'a> {
    pub fn new(device: &'a Device, m: u32, n: u32, k: u32) -> Self {
        Self {
            device,
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
        }
    }

    pub fn run(&self, kernel: &impl GemmKernel) -> Result<()> {
        println!("\n--------------------------------------------------");
        println!("Testing Kernel: {}", kernel.name());
        println!("Matrix size: M={} N={} K={}", self.m, self.n, self.k);

        // Total steps: 1 (Setup) + 3 (Warm-up) + 1 (Benchmark) + 1 (Validation) = 6
        let total_steps = 6;
        let pb = ProgressBar::new(total_steps);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{bar:30.cyan/blue}] {pos}/{len} | {msg}",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.enable_steady_tick(Duration::from_millis(80));

        // Host Memory
        pb.set_message("Allocating host & device buffers...");
        let host_a = vec![1.0f32; (self.m * self.k) as usize];
        let host_b = vec![2.0f32; (self.k * self.n) as usize];
        let mut host_c = vec![0.0f32; (self.m * self.n) as usize];

        // Device Buffer
        let buf_a = create_buffer(&host_a).on(GpuOnly).with(self.device)?;
        let buf_b = create_buffer(&host_b).on(GpuOnly).with(self.device)?;
        let mut buf_c = create_buffer(&host_c).on(GpuOnly).with(self.device)?;
        pb.inc(1);

        // 1. Warm-up pass
        for i in 0..3 {
            pb.set_message(format!("Warm-up pass ({}/3)...", i + 1));
            kernel.launch(
                self.device,
                &buf_a,
                &buf_b,
                &mut buf_c,
                self.m,
                self.n,
                self.k,
                self.alpha,
                self.beta,
            )?;
            self.device.wait_timeline(self.device.dispatch()?)?;
            pb.inc(1);
        }

        // 2. Benchmark
        pb.set_message("Running timed GPU benchmark...");
        let timer = GpuTimer::new(self.device)?;

        let s0 = timer.stamp(self.device);
        for _ in 0..3 {
            kernel.launch(
                self.device,
                &buf_a,
                &buf_b,
                &mut buf_c,
                self.m,
                self.n,
                self.k,
                self.alpha,
                self.beta,
            )?;
        }
        let s1 = timer.stamp(self.device);
        self.device.wait_timeline(self.device.dispatch()?)?;
        pb.inc(1);

        // 3. Compute Validation
        pb.set_message("Reading back & validating results...");
        read_buffer(&buf_c).into(&mut host_c).with(self.device)?;

        let expected = (self.k as f32) * 2.0 * self.alpha;
        let total_len = (self.m * self.n) as usize;

        let check_points = [
            ("Start [0]", 0),
            ("Mid", total_len / 2),
            ("End", total_len - 1),
        ];

        for (label, idx) in check_points {
            let actual = host_c[idx];
            let diff = (actual - expected).abs();

            if diff >= 1e-2 {
                pb.finish_and_clear();
                panic!(
                    "\n[FAIL] Mismatch at {} (index {})! Got {}, expected {}",
                    label, idx, actual, expected
                );
            }
        }
        pb.inc(1);

        pb.finish_and_clear();

        let t0 = timer.fetch(self.device, s0)?;
        let t1 = timer.fetch(self.device, s1)?;
        let avg_dur = (t1 - t0) / 3;
        let ops_per_gemm = 2.0 * (self.m as f64) * (self.n as f64) * (self.k as f64);
        let gflops = (ops_per_gemm / avg_dur.as_secs_f64()) / 1e9;

        println!("Execution time (avg): {:?}", avg_dur);
        println!("Performance: {:.2} GFLOPS", gflops);

        println!(
            "Validation: PASSED (Sample C[0]: {:.2}, Expected: {:.2})",
            host_c[0], expected
        );

        Ok(())
    }
}

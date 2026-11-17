mod kernels;
use kernels::*;

use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

// use gpgpru::config::*;
use gpgpru::kernel;
use gpgpru::prelude::*;
use gpgpru::utils::*;
use gpgpru::{Device, GpuSlice, create_buffer, read_buffer};

/// Simple `clap` cli struct.
#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Subcommands for selecting a kernel.
#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a naively implemented GEMM kernel.
    Naive,
    /// Run a coalesced GEMM kernel.
    Coale,
    /// Run a Shared Memory Cache-Block kernel.
    Block,
    /// Run a Shared Memory Tiled 1D kernel.
    Tile1d,
    /// Run a Shared Memory Tiled 2D kernel.
    Tile2d,
    /// Run a vectorized access tiled kernel.
    Vected,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let device = Device::new()?;
    // let bench = GemmBenchmark::new(&device, 2048, 2048, 2048);
    let bench = GemmBenchmark::new(&device, 4096, 4096, 4096);

    match cli.command {
        Commands::Naive => bench.run(&NaiveKernel::new())?,
        Commands::Coale => bench.run(&CoalescedKernel::new())?,
        Commands::Block => bench.run(&BlockKernel::new())?,
        Commands::Tile1d => bench.run(&Tile1DKernel::new())?,
        Commands::Tile2d => bench.run(&Tile2DKernel::new())?,
        Commands::Vected => bench.run(&VectedKernel::new())?,
    }

    Ok(())
}

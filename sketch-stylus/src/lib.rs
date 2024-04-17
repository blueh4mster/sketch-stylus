#![cfg_attr(not(feature = "export-abi"), no_main)]
extern crate alloc;

mod constants;
mod matrix_op;
mod nn;
mod training;
mod prediction;
/// Use an efficient WASM allocator.
#[global_allocator]
static ALLOC: mini_alloc::MiniAlloc = mini_alloc::MiniAlloc::INIT;

//entrypoint
pub use training::*;

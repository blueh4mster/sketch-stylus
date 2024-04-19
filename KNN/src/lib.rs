//!
//! Stylus Hello World
//!
//! The following contract implements the Counter example from Foundry.
//!
//! ```
//! contract Counter {
//!     uint256 public number;
//!     function setNumber(uint256 newNumber) public {
//!         number = newNumber;
//!     }
//!     function increment() public {
//!         number++;
//!     }
//! }
//! ```
//!
//! The program is ABI-equivalent with Solidity, which means you can call it from both Solidity and Rust.
//! To do this, run `cargo stylus export-abi`.
//!
//! Note: this code is a template-only and has not been audited.
//!

// Allow `cargo stylus export-abi` to generate a main function.
#![cfg_attr(not(feature = "export-abi"), no_main)]
extern crate alloc;

/// Use an efficient WASM allocator.
#[global_allocator]
static ALLOC: mini_alloc::MiniAlloc = mini_alloc::MiniAlloc::INIT;

use sqrt_rs::babylonian_sqrt;
/// Import items from the SDK. The prelude contains common traits and macros.
use stylus_sdk::{alloy_primitives::U256, prelude::*};
// Define some persistent storage using the Solidity ABI.
// `Counter` will be the entrypoint.
sol_storage! {
    #[entrypoint]
    pub struct KNN{
        uint256 k;
    }
}

/// Declare that `Counter` is a contract with the following external methods.

impl KNN {
    pub fn euclidean_distance(x1: Vec<i128>, x2: Vec<i128>) -> i128 {
        // distance = np.sqrt(np.sum((x1-x2)**2))
        assert_eq!(x1.len(), x2.len(), "length of arrays not same");
        let mut sum = 0.0;
        for i in 0..x1.len() {
            let val = (x1[i] - x2[i]) as f32;
            sum += val * val;
        }
        //scale it in here
        let mut ans = babylonian_sqrt(sum);
        ans *= 1000.0; // scaled 10**3 times
        return ans as i128;
    }
}

#[external]
impl KNN {
    pub fn train_predict(&mut self) {}
    pub fn set_k(&mut self, val: U256) {
        self.k.set(val);
    }
    pub fn get_k(&self) -> U256 {
        self.k.get()
    }
}

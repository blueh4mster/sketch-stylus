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

use std::collections::HashMap;
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
    pub fn euclidean_distance(&self, x1: Vec<i128>, x2: Vec<i128>) -> i128 {
        // distance = np.sqrt(np.sum((x1-x2)**2))
        assert_eq!(x1.len(), x2.len(), "length of arrays not same");
        let mut sum = 0;
        for i in 0..x1.len() {
            let val = x1[i] - x2[i];
            sum += val * val;
        }
        //scale it in here
        let mut ans = self.sqrt(sum as u128);
        ans *= 1000; // scaled 10**3 times
        return ans as i128;
    }

    pub fn sqrt(&self,x: u128) -> u128 {
        if x == 0 {
            return x;
        }
    
        let mut x_aux = x;
        let mut result = 1 as u128;
    
        if x_aux >= 0x100000000 {
            x_aux >>= 32;
            result <<= 16;
        }
        if x_aux >= 0x10000 {
            x_aux >>= 16;
            result <<= 8;
        }
        if x_aux >= 0x100 {
            x_aux >>= 8;
            result <<= 4;
        }
        if x_aux >= 0x10 {
            x_aux >>= 4;
            result <<= 2;
        }
        if x_aux >= 0x4 {
            result <<= 1;
        }
        result = (result + x / result) >> 1;
        result = (result + x / result) >> 1;
        result = (result + x / result) >> 1;
        result = (result + x / result) >> 1;
        result = (result + x / result) >> 1;
        result = (result + x / result) >> 1;
        result = (result + x / result) >> 1; 
        let rounded_result = x / result;
        let mut final_res;
    
        if result >= rounded_result {
            final_res = rounded_result;
        } else {
            final_res = result;
        }
        final_res
    }

    pub fn most_common(&self,x:Vec<i128>, x_train : Vec<Vec<i128>>, y_train:Vec<i128>,k:u128) -> i128 {
        let mut distances : Vec<i128> = Vec::new();
        let tmp_len = x_train.len();
        for i in 0..tmp_len{
            let x_clone = x.clone();
            let x_train_clone = x_train[i].clone();
            distances[i] = self.euclidean_distance(x_clone, x_train_clone);
        }
        let mut tmp : Vec<(usize,i128)> = Vec::new();
        let dist_len = distances.len();
        for x in 0..dist_len{
            tmp[x].0 = x;
            tmp[x].1 = distances[x];
        }
        let ki  = k as usize;
        tmp.sort_by(|a,b| a.1.cmp(&b.1));
        let mut k_indices : Vec<usize> = Vec::new();
        for y in 0..ki{
            k_indices[y] = tmp[y].0;
        }
        let mut k_nearest_labels : Vec<i128> = Vec::new();
        for z in 0..ki{
            k_nearest_labels[z] = y_train[k_indices[z]];
        }
        let mut freq_vec: HashMap<i128, u128> = HashMap::new();

        for i in &k_nearest_labels {
            let freq: &mut u128 = freq_vec.entry(*i).or_insert(0);
            *freq += 1;
        }
        let mut start = k_nearest_labels[0];
        let mut maximum_freq = freq_vec[&start];
        for i in 0..ki{
            let i_tmp = k_nearest_labels[i];
            if freq_vec[&i_tmp] > maximum_freq {
                maximum_freq = freq_vec[&i_tmp];
                start = i_tmp;
            }
        }
        start
    }

    pub fn predict(&self, x: Vec<Vec<i128>>, x_train : Vec<Vec<i128>>, y_train:Vec<i128>,k:u128) -> Vec<i128>{
        let mut predictions : Vec<i128> = Vec::new();
        let tmp_len = x.len();
        for i in 0..tmp_len{
            let x_clone = x[i].clone();
            let x_train_clone = x_train.clone();
            let y_train_clone = y_train.clone();
            predictions[i] = self.most_common(x_clone, x_train_clone, y_train_clone, k);
        }
        predictions
    }
}

#[external]
impl KNN {
    pub fn train_predict(&mut self, x: Vec<Vec<i128>>, x_train : Vec<Vec<i128>>, y_train:Vec<i128>,k:u128) {
        self.predict(x, x_train, y_train, k);
    }
    pub fn set_k(&mut self, val: U256) {
        self.k.set(val);
    }
    pub fn get_k(&self) -> U256 {
        self.k.get()
    }

}

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

/// Import items from the SDK. The prelude contains common traits and macros.
use stylus_sdk::prelude::*;
mod functions;
// Define some persistent storage using the Solidity ABI.
// `Counter` will be the entrypoint.
use functions::Functions;
sol_storage! {
    #[entrypoint]
    pub struct LogReg{
        int[][] w;// weights
        int[][] b;// biases
    }
}

impl LogReg {
    pub fn descent(
        &mut self,
        x_train: Vec<Vec<i128>>,
        y_train: Vec<Vec<i128>>,
        iterations: u128,
        lr: i128,
    ) {
        // n_samples, n_features = X.shape
        let n_samples = x_train.len() as i128;
        // let n_features = x_train[0].len() as i128;
        for _ in 0..iterations {
            let x_train_clone = x_train.clone();
            let x_train_clone_clone = x_train_clone.clone();
            let x_train_trans = Functions::transpose(x_train_clone_clone);
            let y_train_clone = y_train.clone();

            let (w, b) = self.change_type();
            let (w_clone, b_clone) = (w.clone(), b.clone());

            let linear_predictions = Functions::sum(Functions::dot_product(x_train_clone, w), b);

            let predictions = Functions::sigmoid_mat(linear_predictions);

            let pred_y = Functions::sum(predictions, Functions::scalar_mul(y_train_clone, -1));
            let pred_y_clone = pred_y.clone();
            let dw =
                Functions::scalar_div(Functions::dot_product(x_train_trans, pred_y), n_samples);

            let db = Functions::scalar_div(Functions::element_sum_row(pred_y_clone), n_samples);

            let updated_w = Functions::sum(w_clone, Functions::scalar_mul(dw, lr));
            let updated_b = Functions::sum(b_clone, Functions::scalar_mul(db, lr));

            self.set_vars(updated_w, updated_b);
        }
    }
    pub fn change_type(&self) -> (Vec<Vec<i128>>, Vec<Vec<i128>>) {
        // initiate new w,b as data types don't match and there is no clone function on storage vec
        let mut w: Vec<Vec<i128>> = Vec::new();
        let mut b: Vec<Vec<i128>> = Vec::new();

        //convert them to i128 and then pass their clones

        //convert them to i128 and then pass their clones
        //set w
        let m = self.w.len();
        let n = self.w.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                w[i][j] = self.w.get(i).unwrap().get(j).unwrap().try_into().unwrap();
            }
        }

        //set w1
        let m = self.b.len();
        let n = self.b.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                b[i][j] = self.b.get(i).unwrap().get(j).unwrap().try_into().unwrap();
            }
        }
        (w, b)
    }

    pub fn init_params(&mut self, m: i128, n: i128) {
        let m = m as usize;
        let b: Vec<Vec<i128>> = vec![vec![0; 1]; m];
        let n = n as usize;
        let w: Vec<Vec<i128>> = vec![vec![0; 1]; n];
        self.set_vars(w, b);
    }

    pub fn set_vars(&mut self, w: Vec<Vec<i128>>, b: Vec<Vec<i128>>) {
        //set w and all.
        let m = self.w.len();
        let n = self.w.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                let mut z = self.w.get_mut(i).unwrap();
                let mut p = z.setter(j).unwrap();
                p.set(w[i][j].try_into().unwrap());
            }
        }
        //set b
        let m = self.b.len();
        let n = self.b.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                let mut z = self.b.get_mut(i).unwrap();
                let mut p = z.setter(j).unwrap();
                p.set(b[i][j].try_into().unwrap());
            }
        }
    }
}

#[external]
impl LogReg {
    pub fn train(
        &mut self,
        x_train: Vec<Vec<i128>>,
        y_train: Vec<Vec<i128>>,
        iterations: u128,
        lr: i128,
    ) -> bool {
        // w-> n*1, b-> m*1, x_train-> m*n, y_train->m*1
        // n_samples, n_features = X.shape
        let n_samples = x_train.len() as i128;
        let n_features = x_train[0].len() as i128;
        self.init_params(n_samples, n_features);

        self.descent(x_train, y_train, iterations, lr);
        true
    }
}

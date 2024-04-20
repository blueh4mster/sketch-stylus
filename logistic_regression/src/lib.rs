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
use stylus_sdk::{alloy_primitives::U256, prelude::*};

// Define some persistent storage using the Solidity ABI.
// `Counter` will be the entrypoint.
sol_storage! {
    #[entrypoint]
    pub struct LogReg{
        int[] w;// weights
        int[] b;// biases
    }
}

impl LogReg {
    pub fn change_type(&self) -> (Vec<i128>, Vec<i128>) {
        // initiate new w1,b1,w2,b2 as data types don't match and there is no clone function on storage vec
        let mut w: Vec<i128> = Vec::new();
        let mut b: Vec<i128> = Vec::new();

        //convert them to i128 and then pass their clones
        //set w
        let m = self.w.len();
        for i in 0..m {
            w[i] = self.w.get(i).unwrap().try_into().unwrap();
        }

        //set b
        let m = self.b.len();
        for i in 0..m {
            b[i] = self.b.get(i).unwrap().try_into().unwrap();
        }
        (w, b)
    }

    pub fn dot_product(m1: Vec<Vec<i128>>, m2: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        //say we have scaled vectors as inputs and scale=1000
        assert_eq!(
            m1[0].len(),
            m2.len(),
            "Number of columns in first matrix must be equal to number of rows in second matrix"
        );
        let m = m1.len();
        let n = m2.len();
        let p = m2[0].len();
        let mut result = vec![vec![0; p]; m];

        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    result[i][j] += m1[i][k] * m2[k][j];
                }
                result[i][j] /= 1000;
            }
        }
        //scaled by 1000 only
        result
    }

    pub fn transpose(&self, ori: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        let m = ori.len();
        let n = ori[0].len();
        let mut trans: Vec<Vec<i128>> = Vec::new();

        for i in 0..m {
            for j in 0..n {
                let ori_j_i = ori[j][i];
                trans[i][j] = ori_j_i;
            }
        }
        trans
    }

    // no change if scaled inputs
    pub fn sum(&self, m1: Vec<Vec<i128>>, m2: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        assert_eq!(m1.len(), m2.len(), "Number of rows donot match");
        let n = m1.len();
        let m = m1[0].len();
        let mut result = vec![vec![0; m]; n];
        for i in 0..n {
            for k in 0..m {
                result[i][k] = m1[i][k] + m2[k][0];
            }
        }
        result
    }
    pub fn element_sum_row(&self, z: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        let mut sum = 0;
        let mut result: Vec<Vec<i128>> = Vec::new();
        for i in 0..z.len() {
            for j in 0..z[0].len() {
                sum += z[i][j];
            }
            result[i][0] = sum;
            sum = 0;
        }
        result
    }

    pub fn sigmoid(&self, z: i128) -> i128 {
        let mut ans = exp(z as f32) / (1.0 + exp(z as f32));
        ans *= 1000.0;
        return ans as i128;
    }

    pub fn set_vars(&mut self, w: Vec<i128>, b: Vec<i128>) {
        //set w1
        let m = self.w.len();
        for i in 0..m {
            let mut z = self.w.get_mut(i).unwrap();
            let mut p = z.setter(i).unwrap();
            p.set(w[i].try_into().unwrap());
        }
        //set b
        let m = self.b.len();
        for i in 0..m {
            let mut z = self.b.get_mut(i).unwrap();
            let mut p = z.setter(i).unwrap();
            p.set(b[i].try_into().unwrap());
        }
    }
}
#[external]
impl LogReg {
    pub fn train(&mut self, x_train: Vec<Vec<i128>>, y_train: Vec<Vec<i128>>, iterations: u128) {
        // init  w, b=0;
        // update their state

        // n_samples, n_features = X.shape
        let n_samples = x_train.len();
        let n_features = x_train[0].len();

        // self.weights = np.zeros(n_features)
        // self.bias = 0
        for _ in 0..iterations {
            let (w, b) = self.change_type();
            self.sum(self.dot_product(x_train, w), b)
        }
        // for _ in range(self.n_iters):
        //     linear_pred = np.dot(X, self.weights) + self.bias
        //     predictions = sigmoid(linear_pred)

        //     dw = (1/n_samples) * np.dot(X.T, (predictions - y))
        //     db = (1/n_samples) * np.sum(predictions-y)

        //     self.weights = self.weights - self.lr*dw
        //     self.bias = self.bias - self.lr*db
    }
}

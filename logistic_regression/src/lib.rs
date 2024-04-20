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

use fast_math::exp;
/// Import items from the SDK. The prelude contains common traits and macros.
use stylus_sdk::prelude::*;
// Define some persistent storage using the Solidity ABI.
// `Counter` will be the entrypoint.
sol_storage! {
    #[entrypoint]
    pub struct LogReg{
        int[][] w;// weights
        int[][] b;// biases
    }
}

impl LogReg {
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

    pub fn dot_product(&self, m1: Vec<Vec<i128>>, m2: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
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

    pub fn sigmoid_mat(&self, z: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        let mut result: Vec<Vec<i128>> = Vec::new();
        for i in 0..z.len() {
            for j in 0..z[0].len() {
                result[i][j] = self.sigmoid(z[i][j]);
            }
        }
        result
    }

    pub fn scalar_div(&self, mat: Vec<Vec<i128>>, scalar: i128) -> Vec<Vec<i128>> {
        let mut result: Vec<Vec<i128>> = Vec::new();
        let m = mat.len();
        let n = mat[0].len();
        for i in 0..m {
            for j in 0..n {
                result[i][j] = mat[i][j] / scalar;
            }
        }
        result
    }

    pub fn scalar_mul(&self, mat: Vec<Vec<i128>>, scalar: i128) -> Vec<Vec<i128>> {
        let mut result: Vec<Vec<i128>> = Vec::new();
        let m = mat.len();
        let n = mat[0].len();
        for i in 0..m {
            for j in 0..n {
                result[i][j] = mat[i][j] * scalar;
            }
        }
        result
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
    ) {
        // w-> n*1, b-> m*1, x_train-> m*n, y_train->m*1
        // n_samples, n_features = X.shape
        let n_samples = x_train.len() as i128;
        let n_features = x_train[0].len() as i128;
        self.init_params(n_samples, n_features);
        // self.weights = np.zeros(n_features)
        // self.bias = 0
        for _ in 0..iterations {
            let x_train_clone = x_train.clone();
            let x_train_clone_clone = x_train_clone.clone();
            let x_train_trans = self.transpose(x_train_clone_clone);
            let y_train_clone = y_train.clone();

            let (w, b) = self.change_type();
            let (w_clone, b_clone) = (w.clone(), b.clone());

            let linear_predictions = self.sum(self.dot_product(x_train_clone, w), b);

            let predictions = self.sigmoid_mat(linear_predictions);

            let pred_y = self.sum(predictions, self.scalar_mul(y_train_clone, -1));
            let pred_y_clone = pred_y.clone();
            let dw = self.scalar_div(self.dot_product(x_train_trans, pred_y), n_samples);

            let db = self.scalar_div(self.element_sum_row(pred_y_clone), n_samples);

            let updated_w = self.sum(w_clone, self.scalar_mul(dw, lr));
            let updated_b = self.sum(b_clone, self.scalar_mul(db, lr));

            self.set_vars(updated_w, updated_b);
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

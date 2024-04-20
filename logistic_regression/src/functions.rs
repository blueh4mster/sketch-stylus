use alloc::vec::Vec;
use fast_math::exp;
use stylus_sdk::prelude::*;

sol_storage! {
    pub struct Functions{}
}

impl Functions {
    pub fn scalar_mul(mat: Vec<Vec<i128>>, scalar: i128) -> Vec<Vec<i128>> {
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

    pub fn sigmoid(z: i128) -> i128 {
        let mut ans = exp(z as f32) / (1.0 + exp(z as f32));
        ans *= 1000.0;
        return ans as i128;
    }

    pub fn sigmoid_mat(z: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        let mut result: Vec<Vec<i128>> = Vec::new();
        for i in 0..z.len() {
            for j in 0..z[0].len() {
                let mut ans = exp(z[i][j] as f32) / (1.0 + exp(z[i][j] as f32));
                ans *= 1000.0;
                result[i][j] = ans as i128;
            }
        }
        result
    }

    pub fn scalar_div(mat: Vec<Vec<i128>>, scalar: i128) -> Vec<Vec<i128>> {
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

    // no change if scaled inputs
    pub fn sum(m1: Vec<Vec<i128>>, m2: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
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

    pub fn element_sum_row(z: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
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

    pub fn transpose(ori: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
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
}

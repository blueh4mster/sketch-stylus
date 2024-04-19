use alloc::vec::Vec;
use fast_math::exp;
use stylus_sdk::prelude::*;

sol_storage! {
    pub struct MatrixOp{}
}

impl MatrixOp {
    pub fn relu_derive(m: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        let mut result: Vec<Vec<i128>> = Vec::new();
        for i in 0..m.len() {
            for j in 0..m[0].len() {
                match m[i][j] > 0 {
                    true => result[i][j] = 1000, //scaling for dot_product
                    false => result[i][j] = 0,
                }
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
    pub fn elementwise_mul(m1: Vec<Vec<i128>>, m2: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        assert_eq!(m1.len(), m2.len(), "Number of rows not equal");
        assert_eq!(m1[0].len(), m2[0].len(), "Number of rows not equal");
        let mut result: Vec<Vec<i128>> = Vec::new();
        for i in 0..m1.len() {
            for j in 0..m1[0].len() {
                result[i][j] = m1[i][j] * m2[i][j];
                //if m1 and m2 are scaled by 1000
                result[i][j] /= 1000;
            }
        }
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

    pub fn softmax(z: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        let mut v: Vec<Vec<i128>> = Vec::new();

        // z comes scaled by 1000 , so we need to divide each element by 1000
        let mut sum: f32 = 0.0;
        for i in 0..z.len() {
            // for scaled input
            let mut zi = z[i][0] as f32;
            zi /= 1000.0;

            sum += exp(zi);
        }
        for i in 0..z.len() {
            // for scaled input
            let mut zi = z[i][0] as f32;
            zi /= 1000.0;
            /////
            v[i][0] = (exp(zi / sum) * 1000.0) as i128;
        }
        v
    }

    pub fn relu(z: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        let len = z.len();
        let len2 = z[0].len();
        let mut v: Vec<Vec<i128>> = Vec::new();
        for i in 0..len {
            for j in 0..len2 {
                if z[i][j] > 0 {
                    v[i][j] = z[i][j];
                } else {
                    v[i][j] = 0;
                }
            }
        }
        v
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

    pub fn one_hot(y: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        let m = y.len();
        let n = y[0].len();
        let size = m * n;
        let mut max_val = y[0][0];
        for i in 0..m {
            for j in 0..n {
                if y[i][j] > max_val {
                    max_val = y[i][j];
                }
            }
        }
        // as the input was scaled , max value is actually 1000 times less
        max_val += 1000; // scaled add

        // actually value for vector size
        max_val /= 1000;
        let final_val = max_val as usize;
        let mut one_hot_y = vec![vec![0; final_val]; size]; // np.zeroes--done
        for row in 0..m {
            let col = y[row][0] as usize;
            one_hot_y[row][col] = 1000; //should go scaled
        }
        one_hot_y
    }

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
}

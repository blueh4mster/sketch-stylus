use crate::matrix_op::MatrixOp;
use stylus_sdk::prelude::*;

sol_storage! {
    pub struct NN {

    }
}

impl NN {
    // use this in the contract training and make a general setter function
    pub fn init_params() -> (
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
    ) {
        let w1: Vec<Vec<i128>> = vec![vec![8; 784]; 10];
        let b1: Vec<Vec<i128>> = vec![vec![1; 1]; 10];
        let w2: Vec<Vec<i128>> = vec![vec![2; 10]; 10];
        let b2: Vec<Vec<i128>> = vec![vec![4; 1]; 10];
        // for i in 0..10 {
        //     for j in 0..784 {
        //         w1[i][j] = rand::random();
        //     }
        // }
        // for i in 0..10 {
        //     for j in 0..1 {
        //         b1[i][j] = rand::random();
        //     }
        // }
        // for i in 0..10 {
        //     for j in 0..10 {
        //         w2[i][j] = rand::random();
        //     }
        // }
        // for i in 0..10 {
        //     for j in 0..1 {
        //         b2[i][j] = rand::random();
        //     }
        // }
        return (w1, b1, w2, b2);
    }

    pub fn get_predictions(a2: Vec<Vec<i128>>) -> Vec<Vec<i128>> {
        let mut prediction: Vec<Vec<i128>> = Vec::new();
        let n = a2.len();
        let m = a2[0].len();
        for i in 0..m {
            let mut max = a2[0][i];
            prediction[0][i] = 0;
            for j in 0..n {
                if a2[j][i] > max {
                    max = a2[j][i];
                    prediction[0][i] = j as i128;
                }
            }
        }
        prediction
    }

    pub fn get_accuracy(prediction: Vec<Vec<i128>>, y: Vec<Vec<i128>>) -> i128 {
        let mut accuracy = 0 as i128;
        let m = y.len();
        let n = y[0].len();
        for i in 0..n {
            for j in 0..m {
                if prediction[i][j] == y[i][j] {
                    accuracy += 1;
                }
            }
        }
        accuracy = accuracy / m as i128;
        accuracy
    }
    pub fn forward_prop(
        w1: Vec<Vec<i128>>,
        b1: Vec<Vec<i128>>,
        w2: Vec<Vec<i128>>,
        b2: Vec<Vec<i128>>,
        x: Vec<Vec<i128>>,
    ) -> (
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
    ) {
        let x_clone = x.clone();
        let inter1 = MatrixOp::dot_product(w1, x);
        let z1 = MatrixOp::sum(inter1, b1);
        let z1_clone = z1.clone();
        let a1 = MatrixOp::relu(z1);
        let inter2 = MatrixOp::dot_product(w2, x);
        let z2 = MatrixOp::sum(inter2, b2);
        let z2_clone = z2.clone();
        let a2 = MatrixOp::softmax(z2);
        return (z1, a1, z2, a2);
    }

    pub fn backward_prop(
        z1: Vec<Vec<i128>>,
        a1: Vec<Vec<i128>>,
        z2: Vec<Vec<i128>>,
        a2: Vec<Vec<i128>>,
        w1: Vec<Vec<i128>>,
        w2: Vec<Vec<i128>>,
        x: Vec<Vec<i128>>,
        y: Vec<Vec<i128>>,
    ) -> (
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
    ) {
        let m = (y.len() * y[0].len()) as i128;
        let one_hot_y = MatrixOp::one_hot(y);
        let dz2 = MatrixOp::sum(a2, MatrixOp::scalar_mul(one_hot_y, -1));
        let dw2 = MatrixOp::scalar_div(MatrixOp::dot_product(z2, MatrixOp::transpose(a1)), m);
        let dz2_clone = dz2.clone();
        let db2 = MatrixOp::scalar_div(MatrixOp::element_sum_row(dz2), m);

        let dz1 = MatrixOp::elementwise_mul(
            MatrixOp::dot_product(MatrixOp::transpose(w2), dz2),
            MatrixOp::relu_derive(z1),
        );
        let dz1_clone = dz1.clone();
        let dw1 = MatrixOp::scalar_div(MatrixOp::dot_product(dz1, MatrixOp::transpose(x)), m);
        let db1 = MatrixOp::scalar_div(MatrixOp::element_sum_row(dz1), m);
        (dw1, db1, dw2, db2)
    }

    pub fn update_params(
        w1: Vec<Vec<i128>>,
        b1: Vec<Vec<i128>>,
        w2: Vec<Vec<i128>>,
        b2: Vec<Vec<i128>>,
        dw1: Vec<Vec<i128>>,
        db1: Vec<Vec<i128>>,
        dw2: Vec<Vec<i128>>,
        db2: Vec<Vec<i128>>,
        alpha: i128,
    ) -> (
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
    ) {
        let updated_w1 = MatrixOp::sum(w1, MatrixOp::scalar_mul(dw1, alpha));
        let updated_b1 = MatrixOp::sum(b1, MatrixOp::scalar_mul(db1, alpha));
        let updated_w2 = MatrixOp::sum(w2, MatrixOp::scalar_mul(dw2, alpha));
        let updated_b2 = MatrixOp::sum(b2, MatrixOp::scalar_mul(db2, alpha));
        (updated_w1, updated_b1, updated_w2, updated_b2)
    }
}

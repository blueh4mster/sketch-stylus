use crate::matrix_op::MatrixOp;
use stylus_sdk::prelude::*;

sol_storage! {
    pub struct NN {

    }
}

impl NN {
    pub fn forward_prop(
        w1: &Vec<Vec<f64>>,
        b1: &Vec<Vec<f64>>,
        w2: &Vec<Vec<f64>>,
        b2: &Vec<Vec<f64>>,
        x: &Vec<Vec<f64>>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let inter1 = MatrixOp::dot_product(w1, &x);
        let z1 = MatrixOp::sum(&inter1, b1);
        let a1 = MatrixOp::relu(&z1);
        let inter2 = MatrixOp::dot_product(w2, x);
        let z2 = MatrixOp::sum(&inter2, b2);
        let a2 = MatrixOp::softmax(&z2);
        return (z1, a1, z2, a2);
    }

    pub fn backward_prop(
        z1: &Vec<Vec<f64>>,
        a1: &Vec<Vec<f64>>,
        z2: &Vec<Vec<f64>>,
        a2: &Vec<Vec<f64>>,
        w1: &Vec<Vec<f64>>,
        w2: &Vec<Vec<f64>>,
        x: &Vec<Vec<f64>>,
        y: &Vec<Vec<f64>>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let m = (y.len() * y[0].len()) as f64;
        let one_hot_y = MatrixOp::one_hot(y);
        let dz2 = MatrixOp::sum(a2, &MatrixOp::scalar_mul(&one_hot_y, -1.0));
        let dw2 = MatrixOp::scalar_div(&MatrixOp::dot_product(z2, &MatrixOp::transpose(a1)), m);
        let db2 = MatrixOp::scalar_div(&MatrixOp::element_sum_row(&dz2), m);

        let dz1 = MatrixOp::elementwise_mul(
            &MatrixOp::dot_product(&MatrixOp::transpose(w2), &dz2),
            &MatrixOp::relu_derive(z1),
        );
        let dw1 = MatrixOp::scalar_div(&MatrixOp::dot_product(&dz1, &MatrixOp::transpose(x)), m);
        let db1 = MatrixOp::scalar_div(&MatrixOp::element_sum_row(&dz1), m);
        (dw1, db1, dw2, db2)
    }

    pub fn update_params(
        w1: &Vec<Vec<f64>>,
        b1: &Vec<Vec<f64>>,
        w2: &Vec<Vec<f64>>,
        b2: &Vec<Vec<f64>>,
        dw1: &Vec<Vec<f64>>,
        db1: &Vec<Vec<f64>>,
        dw2: &Vec<Vec<f64>>,
        db2: &Vec<Vec<f64>>,
        alpha: f64,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let updated_w1 = MatrixOp::sum(w1, &MatrixOp::scalar_mul(dw1, alpha));
        let updated_b1 = MatrixOp::sum(b1, &MatrixOp::scalar_mul(db1, alpha));
        let updated_w2 = MatrixOp::sum(w2, &MatrixOp::scalar_mul(dw2, alpha));
        let updated_b2 = MatrixOp::sum(b2, &MatrixOp::scalar_mul(db2, alpha));
        (updated_w1, updated_b1, updated_w2, updated_b2)
    }
}
use crate::constants::{ConstantParams, Constants};
use crate::nn::NN;
use alloy_sol_types::sol;
use stylus_sdk::prelude::*;

sol_storage! {
    pub struct Training{

    }
}

sol! {
    event prediction_tested(uint128 prediction, uint128 label);
}

impl Training {
    pub fn gradient_descent(
        x: &Vec<Vec<f64>>,
        y: &Vec<Vec<f64>>,
        alpha: f64,
        iterations: u128,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (w1, b1, w2, b2) = NN::init_params();
        for _ in 0..iterations {
            let (z1, a1, z2, a2) = NN::forward_prop(&w1, &b1, &w2, &b2, x);
            let (dw1, db1, dw2, db2) = NN::backward_prop(&z1, &a1, &z2, &a2, &w1, &w2, x, y);
            let (w1, b1, w2, b2) =
                NN::update_params(&w1, &b1, &w2, &b2, &dw1, &db1, &dw2, &db2, alpha);
        }
        (w1, b1, w2, b2)
    }
}

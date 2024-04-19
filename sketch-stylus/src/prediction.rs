// use crate::constants::{ConstantParams, Constants};
// use crate::training::Training;
// use stylus_sdk::prelude::*;

// sol_storage! {
//     pub struct Prediction{
//     }
// }

// #[external]
// impl Prediction {
//     pub fn train_predict() {
//         let (x_train, y_train) = Constants::training_data();
//         let (w1, b1, w2, b2) = Training::gradient_descent(&x_train, &y_train, 0.1, 100);
//         let _ = Training::test_predictions(0, &w1, &b1, &w2, &b2);
//     }
// }

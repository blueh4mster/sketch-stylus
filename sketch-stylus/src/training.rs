use crate::constants::{ConstantParams, Constants};
use crate::nn::NN;
use alloy_sol_types::sol;
use stylus_sdk::evm;
use stylus_sdk::prelude::*;

sol_storage! {
    #[entrypoint]
    pub struct Training{
        int[][] w1;
        int[][] b1;
        int[][] w2;
        int[][] b2;
    }
}

sol! {
    event prediction_tested(uint128 prediction, uint128 label);
}

impl Training {
    pub fn gradient_descent(
        &mut self,
        x: Vec<Vec<i128>>,
        y: Vec<Vec<i128>>,
        alpha: i128,
        iterations: u128,
    ) {
        let (w1, b1, w2, b2) = NN::init_params();
        self.set_vars(w1, b1, w2, b2);
        // set the initial values in here
        for _ in 0..iterations {
            let x_clone = x.clone();
            let y_clone = y.clone();
            let (w1, b1, w2, b2) = self.change_type();
            // make clones to pass in functions
            let (w1_clone, w2_clone, b1_clone, b2_clone) =
                (w1.clone(), w2.clone(), b1.clone(), b2.clone());
            let (w1_clone_clone, w2_clone_clone) = (w1.clone(), w2.clone());

            // call the functions
            let (z1, a1, z2, a2) =
                NN::forward_prop(w1_clone, b1_clone, w2_clone, b2_clone, x_clone);

            let x_clone = x.clone();

            let (dw1, db1, dw2, db2) = NN::backward_prop(
                z1,
                a1,
                z2,
                a2,
                w1_clone_clone,
                w2_clone_clone,
                x_clone,
                y_clone,
            );
            let alpha_clone = alpha.clone();
            let (w1, b1, w2, b2) =
                NN::update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha_clone);
            self.set_vars(w1, b1, w2, b2);
        }
        // (w1, b1, w2, b2)
    }

    pub fn make_predictions(
        &self,
        x: Vec<Vec<i128>>,
        w1: Vec<Vec<i128>>,
        b1: Vec<Vec<i128>>,
        w2: Vec<Vec<i128>>,
        b2: Vec<Vec<i128>>,
    ) -> Vec<Vec<i128>> {
        let (_, _, _, a2) = NN::forward_prop(w1, b1, w2, b2, x); //x is scaled by 1000
        let predictions = NN::get_predictions(a2);
        predictions
    }

    pub fn test_predictions(
        &self,
        index: u128,
        w1: Vec<Vec<i128>>,
        b1: Vec<Vec<i128>>,
        w2: Vec<Vec<i128>>,
        b2: Vec<Vec<i128>>,
    ) {
        let (x_train, y_train) = Constants::training_data(); // get scaled them by 1000
        let mut current_img: Vec<Vec<i128>> = Vec::new();
        let size = x_train.len();
        for i in 0..size {
            current_img[i][0] = x_train[i][index as usize];
        }
        let prediction = self.make_predictions(current_img, w1, b1, w2, b2); //current_img is scalled by 1000
        let label = y_train[0][index as usize] / 1000; // y_train is scaled by 1000
        let p = prediction[0][0] as u128;
        let l = label as u128;

        evm::log(prediction_tested {
            prediction: p,
            label: l,
        });
        println!("prediction : {}", prediction[0][0]);
        println!("label : {}", label);
    }

    pub fn set_vars(
        &mut self,
        w1: Vec<Vec<i128>>,
        b1: Vec<Vec<i128>>,
        w2: Vec<Vec<i128>>,
        b2: Vec<Vec<i128>>,
    ) {
        //set w1 and all.
        let m = self.w1.len();
        let n = self.w1.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                let mut z = self.w1.get_mut(i).unwrap();
                let mut p = z.setter(j).unwrap();
                p.set(w1[i][j].try_into().unwrap());
            }
        }
        //set b1 and all.
        let m = self.b1.len();
        let n = self.b1.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                let mut z = self.b1.get_mut(i).unwrap();
                let mut p = z.setter(j).unwrap();
                p.set(b1[i][j].try_into().unwrap());
            }
        }
        // let w1 = self.w1.clone();
        //set w2 and all.
        let m = self.w2.len();
        let n = self.w2.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                let mut z = self.w2.get_mut(i).unwrap();
                let mut p = z.setter(j).unwrap();
                p.set(w2[i][j].try_into().unwrap());
            }
        }
        //set b2 and all.
        let m = self.b2.len();
        let n = self.b2.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                let mut z = self.b2.get_mut(i).unwrap();
                let mut p = z.setter(j).unwrap();
                p.set(b2[i][j].try_into().unwrap());
            }
        }
    }

    pub fn change_type(
        &self,
    ) -> (
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
        Vec<Vec<i128>>,
    ) {
        // initiate new w1,b1,w2,b2 as data types don't match and there is no clone function on storage vec
        let mut w1: Vec<Vec<i128>> = Vec::new();
        let mut b1: Vec<Vec<i128>> = Vec::new();
        let mut w2: Vec<Vec<i128>> = Vec::new();
        let mut b2: Vec<Vec<i128>> = Vec::new();

        //convert them to i128 and then pass their clones
        //set w1
        let m = self.w1.len();
        let n = self.w1.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                w1[i][j] = self.w1.get(i).unwrap().get(j).unwrap().try_into().unwrap();
            }
        }
        //set b1
        let m = self.b1.len();
        let n = self.b1.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                b1[i][j] = self.b1.get(i).unwrap().get(j).unwrap().try_into().unwrap();
            }
        }
        //set w2
        let m = self.w2.len();
        let n = self.w2.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                w2[i][j] = self.w2.get(i).unwrap().get(j).unwrap().try_into().unwrap();
            }
        }
        //set w1
        let m = self.b2.len();
        let n = self.b2.get(0).unwrap().len();
        for i in 0..m {
            for j in 0..n {
                b2[i][j] = self.b2.get(i).unwrap().get(j).unwrap().try_into().unwrap();
            }
        }
        (w1, b1, w2, b2)
    }
}

#[external]
impl Training {
    pub fn train_predict(&mut self) -> bool {
        let (x_train, y_train) = Constants::training_data();
        self.gradient_descent(x_train, y_train, 10, 100); // scale then you have to scale x_train and y_train too
                                                          // initiate new w1,b1,w2,b2 as data types don't match and there is no clone function on storage vec
        let (w1, b1, w2, b2) = self.change_type();
        self.test_predictions(0, w1, b1, w2, b2);
        true
    }
}

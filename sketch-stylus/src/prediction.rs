use crate::constants::{ConstantParams, Constants};
use crate::training::Training;
use stylus_sdk::prelude::*;

sol_storage! {
    #[entrypoint]
    pub struct Prediction{
    }
}
use alloc::vec::Vec;
use sqrt_rs::babylonian_sqrt;
use stylus_sdk::prelude::*;

sol_storage! {
    pub struct Functions {

    }
}

impl Functions {
    pub fn euclidean_distance(x1: Vec<i128>, x2: Vec<i128>) -> i128 {
        // distance = np.sqrt(np.sum((x1-x2)**2))
        assert_eq!(x1.len(), x2.len(), "length of arrays not same");
        let mut sum = 0.0;
        for i in 0..x1.len() {
            let val = (x1[i] - x2[i]) as f32;
            sum += val * val;
        }
        //scale it in here
        let mut ans = babylonian_sqrt(sum);
        ans *= 1000.0; // scaled 10**3 times
        return ans as i128;
    }
}

use csv::Reader;
use std::fs::File;

pub trait ConstantParams {
    fn training_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>);
}

pub struct Constants;

impl ConstantParams for Constants {
    fn training_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let file_path = "./../csv/train.csv";
        let file = File::open(file_path).unwrap();
        let mut rdr = Reader::from_reader(file);
        let mut records = Vec::new();
        for result in rdr.records() {
            let record = result.unwrap();
            records.push(record);
        }

        // let mut rng = rand::thread_rng();
        // records.shuffle(&mut rng);

        let _ = records.len();
        let n = records[0].len();
        let mut x_train = Vec::new();
        let mut y_temp = Vec::new();

        for record in records {
            let mut x = Vec::new();
            for i in 1..n {
                let value: f64 = record[i].parse().unwrap();
                x.push(value / 255.0); // Assuming normalization by 255 as in your Python code
            }
            x_train.push(x);
            let y: f64 = record[0].parse().unwrap();
            y_temp.push(y);
        }
        let y_train = vec![y_temp];

        // let x_train = vec![vec![0.0; 784]; 41000];
        // let y_train = vec![vec![0.0; 785]; 1];

        (x_train, y_train)
    }
}

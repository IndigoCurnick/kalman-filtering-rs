use alternative_non_linear::alternative_non_linear;
use linear_a_priori::linear_a_priori;
use linear_first_order::linear_first_order;
use linear_second_order::linear_second_order;
use non_linear::non_linear;
use non_linear_a_priori::non_linear_a_priori;
use peroxide::prelude::{matrix, Matrix, Shape::Row};
use rand_distr::{Distribution, Normal};

pub const A: f64 = 1.0; // Amplitude
pub const OMEGA: f64 = 5.0; // Frequency
pub const TS: f64 = 0.01; // Time sampling
pub const R: f64 = 0.2;
const LINEAR_FIRST_ORDER_Q: f64 = 1.0;

pub const WRITE: bool = true;

mod alternative_non_linear;
mod linear_a_priori;
mod linear_first_order;
mod linear_second_order;
mod non_linear;
mod non_linear_a_priori;

fn main() {
    linear_first_order();
    linear_second_order();
    linear_a_priori();
    non_linear();
    non_linear_a_priori();
    alternative_non_linear();
}

pub fn get_data() -> Data {
    let mut t_hist = vec![];
    let mut y_hist = vec![];
    let mut y_m = vec![];
    let mut t = 0.0;

    let normal = Normal::new(0.0, R).unwrap();
    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let y = A * (OMEGA * t).sin();

        t_hist.push(t);
        y_hist.push(y);
        y_m.push(y + normal.sample(&mut rng));

        t += TS;
    }

    return Data {
        t: t_hist,
        y: y_hist,
        y_m: y_m,
    };
}

pub struct Data {
    pub t: Vec<f64>,
    pub y: Vec<f64>,
    pub y_m: Vec<f64>,
}

fn q_linear_first_order(dt: f64) -> Matrix {
    return LINEAR_FIRST_ORDER_Q
        * matrix(
            vec![
                dt.powf(3.0) / 3.0,
                dt.powf(2.0) / 2.0,
                dt.powf(2.0) / 2.0,
                dt,
            ],
            2,
            2,
            Row,
        );
}

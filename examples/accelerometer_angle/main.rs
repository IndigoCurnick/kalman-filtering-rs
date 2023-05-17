use std::f64::consts::PI;

use kalman_filtering_rs::{predict, update};
use peroxide::prelude::{eye, matrix, zeros, Shape::Row};
use plotly::{Plot, Scatter};

const B: f64 = 0.1;
const SF: f64 = 0.05;
const K: f64 = 0.01;
const G: f64 = 9.81;

fn main() {
    let data = get_data();

    let mut x = matrix(vec![0.0, 0.0, 0.0], 3, 1, Row);
    let mut p = eye(3);

    let f = eye(3);

    let mut b_history = vec![];
    let mut sf_history = vec![];
    let mut k_history = vec![];

    let q = zeros(3, 3);
    for a in data {
        let z = matrix(vec![a], 1, 1, Row);
        let r = matrix(vec![G.powf(2.0) * a.sin().powf(2.0) * 1.0], 1, 1, Row);
        let h = matrix(vec![1.0, G * a.cos(), (G * a.cos()).powf(2.0)], 1, 3, Row);
        let (new_x, new_cov) = predict(&x, &p, &f, &q, None);
        let (new_x, new_cov) = update(&new_x, &new_cov, &h, &z, &r);

        b_history.push(new_x.data[0]);
        sf_history.push(new_x.data[1]);
        k_history.push(new_x.data[2]);

        x = new_x;
        p = new_cov;
    }

    let mut index = vec![];
    for i in 0..b_history.len() {
        index.push(i);
    }

    let mut plot = Plot::new();

    let b_trace = Scatter::new(index.clone(), b_history).name("B");
    let sf_trace = Scatter::new(index.clone(), sf_history).name("SF");
    let k_trace = Scatter::new(index.clone(), k_history).name("K");

    plot.add_traces(vec![b_trace, sf_trace, k_trace]);

    plot.show();
}

fn get_data() -> Vec<f64> {
    let mut m = vec![];

    let mut x = 0.0;

    while x < 2.0 * PI {
        let a = G * x.cos() + B + SF * G * x.cos() + K * (G * x.cos()).powf(2.0);
        m.push(a);
        x += 0.05;
    }

    return m;
}

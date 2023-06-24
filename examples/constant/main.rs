use kalman_filtering_rs::{make_k, make_m, new_cov};
use peroxide::prelude::{matrix, Shape::Row};
use plotly::{Plot, Scatter};
use rand_distr::{Distribution, Normal};

fn main() {
    let measurements = gen_measurements(0.0, 20, 10.0);

    let mut x = matrix(vec![10.0], 1, 1, Row);
    let mut p = matrix(vec![5.0], 1, 1, Row);

    let h = matrix(vec![1.0], 1, 1, Row);
    let phi = matrix(vec![1.0], 1, 1, Row);
    let q = matrix(vec![0.0], 1, 1, Row);
    let r = matrix(vec![10.0], 1, 1, Row);

    let mut history = vec![];
    for measurement in &measurements {
        let x_star = *measurement;

        let xkminusone = x.data[0];

        let x_tilde = x_star - xkminusone;

        let m = make_m(&phi, &p, &q);
        let k = make_k(&m, &h, &r);

        let k1 = k.data[0];

        let x_hat = xkminusone + k1 * x_tilde;

        x = matrix(vec![x_hat], 1, 1, Row);
        p = new_cov(&k, &h, &m);

        history.push(x.data[0]);
    }

    let mut inex = vec![];

    for i in 0..history.len() {
        inex.push(i);
    }

    let mut plot = Plot::new();
    let h_trace = Scatter::new(inex.clone(), history).name("Filter");
    let m_trace = Scatter::new(inex, measurements).name("Measurements");
    plot.add_trace(h_trace);
    plot.add_trace(m_trace);
    plot.show();
}

fn gen_measurements(t: f64, n: usize, s: f64) -> Vec<f64> {
    let mut out = vec![];
    let normal = Normal::new(0.0, s).unwrap();
    let mut rng = rand::thread_rng();
    for _i in 0..n {
        let r: f64 = normal.sample(&mut rng);
        out.push(t + r);
    }

    return out;
}

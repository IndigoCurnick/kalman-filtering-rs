use kalman_filtering_rs::{make_k, make_m, new_cov};
use peroxide::prelude::{eye, matrix, Shape::Row};
use plotly::{Plot, Scatter};
use rand_distr::{Distribution, Normal};

fn main() {
    let speed = 2.0;
    let measurements = get_data(0.0, speed, 30, 3.0);

    let mut index = vec![];

    let mut x = matrix(vec![3.0, 0.0], 2, 1, Row);
    let mut p = eye(2);

    let h = matrix(vec![1.0, 0.0], 1, 2, Row);
    let f = matrix(vec![1.0, 1.0, 0.0, 1.0], 2, 2, Row);

    let mut position_history = vec![];
    let mut speed_history = vec![];

    let q = 0.1 * matrix(vec![1.0 / 3.0, 1.0 / 2.0, 1.0 / 2.0, 1.0], 2, 2, Row);

    for meas in &measurements {
        let r = matrix(vec![2.0], 1, 1, Row);
        let x_star = *meas;
        let m = make_m(&f, &p, &q);
        let k = make_k(&m, &h, &r);

        let xkminus1 = x.data[0];
        let xdotkminus1 = x.data[1];

        let xdot_bar = xdotkminus1;
        let x_bar = xkminus1 + 1.0 * xdot_bar;

        let k1 = k[(0, 0)];
        let k2 = k[(1, 0)];

        let x_tilde = x_star - x_bar;

        let x_hat = x_bar + k1 * x_tilde;
        let xdot_hat = xdot_bar + k2 * x_tilde;

        position_history.push(x_hat);
        speed_history.push(xdot_hat);

        x = matrix(vec![x_hat, xdot_hat], 2, 1, Row);
        p = new_cov(&k, &h, &m);
    }

    let mut true_speeds = vec![];
    for i in 0..measurements.len() {
        index.push(i);
        true_speeds.push(speed);
    }

    let mut plot = Plot::new();

    let measurement_trace = Scatter::new(index.clone(), measurements).name("Measurement");
    let filter_trace = Scatter::new(index.clone(), position_history).name("Filter");
    plot.add_trace(measurement_trace);
    plot.add_trace(filter_trace);
    plot.show();

    let mut plot = Plot::new();

    let measurement_trace = Scatter::new(index.clone(), true_speeds).name("Speed");
    let filter_trace = Scatter::new(index.clone(), speed_history).name("Filter");
    plot.add_trace(measurement_trace);
    plot.add_trace(filter_trace);
    plot.show();
}

fn get_data(starting_point: f64, velocity: f64, n: usize, s: f64) -> Vec<f64> {
    let mut out = vec![];
    let mut x = starting_point;
    out.push(x);

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, s).unwrap();

    for _i in 0..n {
        x = x + velocity;

        let r = normal.sample(&mut rng);
        out.push(x + r);
    }

    return out;
}

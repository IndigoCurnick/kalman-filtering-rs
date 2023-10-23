use std::ops::Div;

use kalman_filtering_rs::{make_k, make_m, new_cov};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const A: f64 = 1.0; // Amplitude
const OMEGA: f64 = 5.0; // Frequency
const TS: f64 = 0.01; // Time sampling
const R: f64 = 0.2;
const LINEAR_FIRST_ORDER_Q: f64 = 1.0;
const LINEAR_SECOND_ORDER_Q: f64 = 10.0;
const Q1: f64 = 10.0;
const Q2: f64 = 5.0;
const NON_LINEAR_A_PRIORI_Q: f64 = 10.0;
const NON_LINEAR_ALTERNATIVE_Q: f64 = 10.0;

fn main() {
    // linear_first_order();
    // linear_second_order();
    // linear_a_priori();
    // non_linear();
    // non_linear_a_priori();
    alternative_non_linear();
}

fn linear_first_order() {
    let data = get_data();

    let mut state = matrix(vec![0.0, 0.0], 2, 1, Row);

    let mut cov = zeros(2, 2);
    cov[(0, 0)] = 999999.9;
    cov[(1, 1)] = 999999.9;

    let h = matrix(vec![1.0, 0.0], 1, 2, Row);
    let q = q_linear_first_order(TS);
    let r = matrix(vec![R.powf(2.0)], 1, 1, Row);
    let phi = matrix(vec![1.0, TS, 0.0, 1.0], 2, 2, Row);

    let mut x_filter = vec![];

    let mut measurement_residuals = vec![];
    let mut filter_residuals = vec![];

    for i in 0..data.t.len() {
        let x_star = data.y_m[i];

        let m = make_m(&phi, &cov, &q);
        let k = make_k(&m, &h, &r);

        let xkminus1 = state.data[0];
        let xdotkminus1 = state.data[1];

        let x_tilde = x_star - xkminus1 - xdotkminus1 * TS;

        let x_hat = xkminus1 + TS * xdotkminus1 + k[(0, 0)] * x_tilde;
        let x_dot_hat = xdotkminus1 + k[(1, 0)] * x_tilde;

        x_filter.push(x_hat);
        measurement_residuals.push(x_star - data.y[i]);
        filter_residuals.push(x_hat - data.y[i]);

        state = matrix(vec![x_hat, x_dot_hat], 2, 1, Row);
        cov = new_cov(&k, &h, &m);
    }

    // Sin Wave Plot
    let mut full_plot = Plot::new();
    let ideal_trace = Scatter::new(data.t.clone(), data.y.clone()).name("Theory");
    full_plot.add_trace(ideal_trace);

    let measurement_trace = Scatter::new(data.t.clone(), data.y_m.clone()).name("Measurements");
    full_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_filter.clone()).name("Filter");
    full_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Linear First Order"));
    full_plot.set_layout(layout);
    full_plot.show();

    // Residuals
    let mut residual_plot = Plot::new();
    let measurement_trace =
        Scatter::new(data.t.clone(), measurement_residuals).name("Measurements");
    residual_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), filter_residuals).name("Filter");
    residual_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Linear First Order Residuals"));
    residual_plot.set_layout(layout);
    residual_plot.show();
}

fn linear_second_order() {
    let data = get_data();

    let mut state = matrix(vec![0.0, 0.0, 0.0], 3, 1, Row);
    let mut cov = zeros(3, 3);
    cov[(0, 0)] = 999999.9;
    cov[(1, 1)] = 999999.9;
    cov[(2, 2)] = 999999.9;

    let h = matrix(vec![1.0, 0.0, 0.0], 1, 3, Row);
    let q = q_linear_second_order(TS);
    let r = matrix(vec![R.powf(2.0)], 1, 1, Row);
    let phi = matrix(
        vec![1.0, TS, 0.5 * TS.powf(2.0), 0.0, 1.0, TS, 0.0, 0.0, 1.0],
        3,
        3,
        Row,
    );

    let mut filter_history = vec![];

    let mut measurement_residuals = vec![];
    let mut filter_residuals = vec![];

    for i in 0..data.t.len() {
        let x_star = data.y_m[i];

        let m = make_m(&phi, &cov, &q);
        let k = make_k(&m, &h, &r);

        let xkminus1 = state.data[0];
        let xdotkminus1 = state.data[1];
        let xdotdotkminus1 = state.data[2];

        let x_tilde = x_star - xkminus1 - TS * xdotkminus1 - 0.5 * TS.powf(2.0) * xdotdotkminus1;

        let k1 = k[(0, 0)];
        let k2 = k[(1, 0)];
        let k3 = k[(2, 0)];

        let x_hat =
            xkminus1 + TS * xdotkminus1 + 0.5 * TS.powf(2.0) * xdotdotkminus1 + k1 * x_tilde;
        let x_dot_hat = xdotkminus1 + TS * xdotdotkminus1 + k2 * x_tilde;
        let x_dot_dot_hat = xdotdotkminus1 + k3 * x_tilde;

        filter_history.push(x_hat);
        filter_residuals.push(x_hat - data.y[i]);
        measurement_residuals.push(x_star - data.y[i]);

        state = matrix(vec![x_hat, x_dot_hat, x_dot_dot_hat], 3, 1, Row);
        cov = new_cov(&k, &h, &m);
    }

    // Sin Wave Plot
    let mut full_plot = Plot::new();
    let ideal_trace = Scatter::new(data.t.clone(), data.y.clone()).name("Theory");
    full_plot.add_trace(ideal_trace);

    let measurement_trace = Scatter::new(data.t.clone(), data.y_m.clone()).name("Measurements");
    full_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), filter_history.clone()).name("Filter");
    full_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Linear Second Order"));
    full_plot.set_layout(layout);
    full_plot.show();

    // Residuals
    let mut residual_plot = Plot::new();
    let measurement_trace =
        Scatter::new(data.t.clone(), measurement_residuals).name("Measurements");
    residual_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), filter_residuals).name("Filter");
    residual_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Linear Second Order Residuals"));
    residual_plot.set_layout(layout);
    residual_plot.show();
}

fn linear_a_priori() {
    let data = get_data();

    let mut state = matrix(vec![0.0, 0.0], 2, 1, Row);

    let mut cov = zeros(2, 2);
    cov[(0, 0)] = 999999.9;
    cov[(1, 1)] = 999999.9;

    let h = matrix(vec![1.0, 0.0], 1, 2, Row);
    let q = q_linear_first_order(TS);
    let r = matrix(vec![R.powf(2.0)], 1, 1, Row);
    let phi = matrix(
        vec![
            (OMEGA * TS).cos(),
            (OMEGA * TS).sin() / OMEGA,
            -OMEGA * (OMEGA * TS).sin(),
            (OMEGA * TS).cos(),
        ],
        2,
        2,
        Row,
    );

    let mut x_filter = vec![];

    let mut measurement_residuals = vec![];
    let mut filter_residuals = vec![];

    for i in 0..data.t.len() {
        let x_star = data.y_m[i];

        let m = make_m(&phi, &cov, &q);
        let k = make_k(&m, &h, &r);

        let xkminus1 = state.data[0];
        let xdotkminus1 = state.data[1];

        let k1 = k[(0, 0)];
        let k2 = k[(1, 0)];

        let x_tilde =
            x_star - (OMEGA * TS).cos() * xkminus1 - ((OMEGA * TS).sin() / OMEGA) * xdotkminus1;

        let x_hat = xkminus1 * (OMEGA * TS).cos()
            + (((OMEGA * TS).sin()) / OMEGA) * xdotkminus1
            + k1 * x_tilde;
        let x_dot_hat = -OMEGA * (OMEGA * TS).sin() * xkminus1
            + (OMEGA * TS).cos() * xdotkminus1
            + k2 * x_tilde;

        x_filter.push(x_hat);
        measurement_residuals.push(x_star - data.y[i]);
        filter_residuals.push(x_hat - data.y[i]);

        state = matrix(vec![x_hat, x_dot_hat], 2, 1, Row);
        cov = new_cov(&k, &h, &m);
    }

    // Sin Wave Plot
    let mut full_plot = Plot::new();
    let ideal_trace = Scatter::new(data.t.clone(), data.y.clone()).name("Theory");
    full_plot.add_trace(ideal_trace);

    let measurement_trace = Scatter::new(data.t.clone(), data.y_m.clone()).name("Measurements");
    full_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_filter.clone()).name("Filter");
    full_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Linear A Priori"));
    full_plot.set_layout(layout);
    full_plot.show();

    // Residuals
    let mut residual_plot = Plot::new();
    let measurement_trace =
        Scatter::new(data.t.clone(), measurement_residuals).name("Measurements");
    residual_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), filter_residuals).name("Filter");
    residual_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Linear A Priori Residuals"));
    residual_plot.set_layout(layout);
    residual_plot.show();
}

fn non_linear() {
    let data = get_data();
    let mut state = matrix(vec![1.0, OMEGA, A], 3, 1, Row);
    let mut cov = zeros(3, 3);
    cov[(0, 0)] = 999999.9;
    cov[(1, 1)] = 999999.9;
    cov[(2, 2)] = 999999.9;

    let q = q_non_linear(TS);
    let r = matrix(vec![R.powf(2.0)], 1, 1, Row);
    let phi = matrix(vec![1.0, TS, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 3, Row);

    let mut x_from_phi_history = vec![];
    let mut x_from_omega_history = vec![];

    let mut measurement_residuals = vec![];
    let mut x_from_phi_residuals = vec![];
    let mut x_from_omega_residuals = vec![];

    for i in 0..data.t.len() {
        let x_star = data.y_m[i];

        let phi_kminus1 = state.data[0];
        let omega_kminus1 = state.data[1];
        let a_kminus1 = state.data[2];

        let phi_bar = phi_kminus1 + omega_kminus1 * TS;
        let omega_bar = omega_kminus1;
        let a_bar = a_kminus1;

        let h = matrix(vec![a_bar * phi_bar.cos(), 0.0, phi_bar.sin()], 1, 3, Row);

        let m = make_m(&phi, &cov, &q);
        let k = make_k(&m, &h, &r);

        let k1 = k[(0, 0)];
        let k2 = k[(1, 0)];
        let k3 = k[(2, 0)];

        let x_tilde = x_star - a_bar * phi_bar.sin();

        let phi_hat = phi_bar + k1 * x_tilde;
        let omega_hat = omega_bar + k2 * x_tilde;
        let a_hat = a_bar + k3 * x_tilde;

        let x_from_phi = a_hat * phi_hat.sin();
        let x_from_omega = a_hat * (omega_hat * data.t[i]).sin();

        x_from_phi_history.push(x_from_phi);
        x_from_omega_history.push(x_from_omega);

        measurement_residuals.push(x_star - data.y[i]);
        x_from_phi_residuals.push(x_from_phi - data.y[i]);
        x_from_omega_residuals.push(x_from_omega - data.y[i]);

        state = matrix(vec![phi_hat, omega_hat, a_hat], 3, 1, Row);
        cov = new_cov(&k, &h, &m);
    }

    // Sin Wave Plot
    let mut full_plot = Plot::new();
    let ideal_trace = Scatter::new(data.t.clone(), data.y.clone()).name("Theory");
    full_plot.add_trace(ideal_trace);

    let measurement_trace = Scatter::new(data.t.clone(), data.y_m.clone()).name("Measurements");
    full_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_from_phi_history).name("x from phi");
    full_plot.add_trace(filter_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_from_omega_history).name("x from omega");
    full_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Non-Linear"));
    full_plot.set_layout(layout);
    full_plot.show();

    // Residuals
    let mut residual_plot = Plot::new();
    let measurement_trace =
        Scatter::new(data.t.clone(), measurement_residuals).name("Measurements");
    residual_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_from_omega_residuals).name("x from omega");
    residual_plot.add_trace(filter_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_from_phi_residuals).name("x from phi");
    residual_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Non-Linear Residuals"));
    residual_plot.set_layout(layout);
    residual_plot.show();
}

fn non_linear_a_priori() {
    let data = get_data();
    let mut state = matrix(vec![1.0, OMEGA], 2, 1, Row);
    let mut cov = zeros(2, 2);
    cov[(0, 0)] = 999999.9;
    cov[(1, 1)] = 999999.9;

    let q = q_non_linear_a_priori(TS);
    let r = matrix(vec![R.powf(2.0)], 1, 1, Row);
    let phi = matrix(vec![1.0, TS, 0.0, 1.0], 2, 2, Row);

    let mut x_from_phi_history = vec![];
    let mut x_from_omega_history = vec![];

    let mut measurement_residuals = vec![];
    let mut x_from_phi_residuals = vec![];
    let mut x_from_omega_residuals = vec![];

    for i in 0..data.t.len() {
        let x_star = data.y_m[i];

        let phi_kminus1 = state.data[0];
        let omega_kminus1 = state.data[1];

        let phi_bar = phi_kminus1 + omega_kminus1 * TS;
        let omega_bar = omega_kminus1;

        let h = matrix(vec![A * phi_bar.cos(), 0.0], 1, 2, Row);

        let m = make_m(&phi, &cov, &q);
        let k = make_k(&m, &h, &r);

        let k1 = k[(0, 0)];
        let k2 = k[(1, 0)];

        let x_tilde = x_star - A * phi_bar.sin();

        let phi_hat = phi_bar + k1 * x_tilde;
        let omega_hat = omega_bar + k2 * x_tilde;

        let x_from_phi = A * phi_hat.sin();
        let x_from_omega = A * (omega_hat * data.t[i]).sin();

        x_from_phi_history.push(x_from_phi);
        x_from_omega_history.push(x_from_omega);

        measurement_residuals.push(x_star - data.y[i]);
        x_from_phi_residuals.push(x_from_phi - data.y[i]);
        x_from_omega_residuals.push(x_from_omega - data.y[i]);

        state = matrix(vec![phi_hat, omega_hat], 2, 1, Row);
        cov = new_cov(&k, &h, &m);
    }

    // Sin Wave Plot
    let mut full_plot = Plot::new();
    let ideal_trace = Scatter::new(data.t.clone(), data.y.clone()).name("Theory");
    full_plot.add_trace(ideal_trace);

    let measurement_trace = Scatter::new(data.t.clone(), data.y_m.clone()).name("Measurements");
    full_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_from_phi_history).name("x from phi");
    full_plot.add_trace(filter_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_from_omega_history).name("x from omega");
    full_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Non-Linear A Priori"));
    full_plot.set_layout(layout);
    full_plot.show();

    // Residuals
    let mut residual_plot = Plot::new();
    let measurement_trace =
        Scatter::new(data.t.clone(), measurement_residuals).name("Measurements");
    residual_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_from_omega_residuals).name("x from omega");
    residual_plot.add_trace(filter_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_from_phi_residuals).name("x from phi");
    residual_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Non-Linear A Priori Residuals"));
    residual_plot.set_layout(layout);
    residual_plot.show();
}

fn alternative_non_linear() {
    fn project(x: f64, x_dot: f64, omega: f64, step: f64) -> (f64, f64) {
        let mut x_bar = x;
        let mut x_dot_bar = x_dot;
        let mut t = 0.0;
        while t <= TS {
            let x_dot_dot = -omega.powf(2.0) * x_bar;
            x_dot_bar = x_dot_bar + step * x_dot_dot;
            x_bar = x_bar + step * x_dot_bar;
            t = t + step;
        }

        return (x_bar, x_dot_bar);
    }

    let data = get_data();
    let mut state = matrix(vec![0.0, 0.0, OMEGA], 3, 1, Row);
    let mut cov = zeros(3, 3);
    cov[(0, 0)] = 999999.9;
    cov[(1, 1)] = 999999.9;
    cov[(2, 2)] = 999999.9;

    let r = matrix(vec![R.powf(2.0)], 1, 1, Row);
    let h = matrix(vec![1.0, 0.0, 0.0], 1, 3, Row);

    let mut x_history = vec![];

    let mut measurement_residuals = vec![];
    let mut filter_residuals = vec![];

    for i in 0..data.t.len() {
        let xkminus1 = state.data[0];
        let xdotkminus1 = state.data[1];
        let omegaminus1 = state.data[2];

        let phi = matrix(
            vec![
                1.0,
                TS,
                0.0,
                -omegaminus1 * TS,
                1.0,
                -2.0 * omegaminus1 * xkminus1 * TS,
                0.0,
                0.0,
                1.0,
            ],
            3,
            3,
            Row,
        );
        let q = q_alternative_non_linear(TS, omegaminus1, xkminus1);

        let m = make_m(&phi, &cov, &q);
        let k = make_k(&m, &h, &r);

        let (x_bar, x_dot_bar) = project(xkminus1, xdotkminus1, omegaminus1, 1e-5);

        let x_star = data.y_m[i];

        let x_tilde = x_star - x_bar;

        let k1 = k[(0, 0)];
        let k2 = k[(1, 0)];
        let k3 = k[(2, 0)];

        let x_hat = x_bar + k1 * x_tilde;
        let x_dot_hat = x_dot_bar + k2 * x_tilde;
        let omega_hat = omegaminus1 + k3 * x_tilde;

        x_history.push(x_hat);
        filter_residuals.push(x_hat - data.y[i]);
        measurement_residuals.push(x_star - data.y[i]);

        state = matrix(vec![x_hat, x_dot_hat, omega_hat], 3, 1, Row);
        cov = new_cov(&k, &h, &m);
    }

    // Sin Wave Plot
    let mut full_plot = Plot::new();
    let ideal_trace = Scatter::new(data.t.clone(), data.y.clone()).name("Theory");
    full_plot.add_trace(ideal_trace);

    let measurement_trace = Scatter::new(data.t.clone(), data.y_m.clone()).name("Measurements");
    full_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), x_history).name("x");
    full_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Non-Linear Alternative"));
    full_plot.set_layout(layout);
    full_plot.show();

    // Residuals
    let mut residual_plot = Plot::new();
    let measurement_trace =
        Scatter::new(data.t.clone(), measurement_residuals).name("Measurements");
    residual_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(data.t.clone(), filter_residuals).name("x");
    residual_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Non-Linear Alternative Residuals"));
    residual_plot.set_layout(layout);
    residual_plot.show();
}

fn q_non_linear(dt: f64) -> Matrix {
    return matrix(
        vec![
            (Q1 * dt.powf(3.0)) / 3.0,
            (Q1 * dt.powf(2.0)) / 2.0,
            0.0,
            (Q1 * dt.powf(2.0)) / 2.0,
            Q1 * dt,
            0.0,
            0.0,
            0.0,
            Q2 * dt,
        ],
        3,
        3,
        Row,
    );
}

fn q_non_linear_a_priori(dt: f64) -> Matrix {
    return NON_LINEAR_A_PRIORI_Q
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

fn q_linear_second_order(dt: f64) -> Matrix {
    return LINEAR_SECOND_ORDER_Q
        * matrix(
            vec![
                dt.powf(5.0) / 20.0,
                dt.powf(4.0) / 8.0,
                dt.powf(3.0) / 6.0,
                dt.powf(4.0) / 8.0,
                dt.powf(3.0) / 3.0,
                dt.powf(2.0) / 2.0,
                dt.powf(3.0) / 6.0,
                dt.powf(2.0) / 2.0,
                dt,
            ],
            3,
            3,
            Row,
        );
}

fn q_alternative_non_linear(dt: f64, omega: f64, x: f64) -> Matrix {
    return NON_LINEAR_ALTERNATIVE_Q
        * matrix(
            vec![
                0.0,
                0.0,
                0.0,
                0.0,
                4.0.div(3.0) * x.powf(2.0) * omega.powf(2.0) * dt.powf(3.0),
                -omega * x * dt.powf(2.0),
                0.0,
                -omega * x * dt.powf(2.0),
                dt,
            ],
            3,
            3,
            Row,
        );
}

fn get_data() -> Data {
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

struct Data {
    pub t: Vec<f64>,
    pub y: Vec<f64>,
    pub y_m: Vec<f64>,
}

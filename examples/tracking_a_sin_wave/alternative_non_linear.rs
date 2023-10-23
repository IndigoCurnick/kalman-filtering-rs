use std::ops::Div;

use kalman_filtering_rs::{make_k, make_m, new_cov};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, Layout, Plot, Scatter};

use crate::{get_data, OMEGA, R, TS};

const NON_LINEAR_ALTERNATIVE_Q: f64 = 10.0;

pub fn alternative_non_linear() {
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

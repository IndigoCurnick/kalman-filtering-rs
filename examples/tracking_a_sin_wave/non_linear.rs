use kalman_filtering_rs::{make_k, make_m, new_cov, write_to_file};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, Layout, Plot, Scatter};

use crate::{get_data, A, OMEGA, R, TS, WRITE};

const Q1: f64 = 10.0;
const Q2: f64 = 5.0;

pub fn non_linear() {
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

    if WRITE {
        let namespace = "non-linear".to_string();
        write_to_file(
            &format!("full-plot-{}.html.tera", namespace),
            &full_plot.to_inline_html("full-plot-non-linear"),
        );
        write_to_file(
            &format!("residual-{}.html.tera", namespace),
            &residual_plot.to_inline_html("residual-non-linear"),
        );
    }
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

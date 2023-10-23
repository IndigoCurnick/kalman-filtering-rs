use kalman_filtering_rs::{make_k, make_m, new_cov};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, Layout, Plot, Scatter};

use crate::{get_data, A, OMEGA, R, TS};

const NON_LINEAR_A_PRIORI_Q: f64 = 10.0;

pub fn non_linear_a_priori() {
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

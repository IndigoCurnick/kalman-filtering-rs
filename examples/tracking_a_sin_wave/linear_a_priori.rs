use kalman_filtering_rs::{make_k, make_m, new_cov};
use peroxide::prelude::{matrix, zeros, Shape::Row};
use plotly::{common::Title, Layout, Plot, Scatter};

use crate::{get_data, q_linear_first_order, OMEGA, R, TS};

pub fn linear_a_priori() {
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

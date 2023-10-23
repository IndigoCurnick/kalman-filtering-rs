use kalman_filtering_rs::{make_k, make_m, new_cov};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, Layout, Plot, Scatter};

use crate::{get_data, R, TS};

const LINEAR_SECOND_ORDER_Q: f64 = 10.0;

pub fn linear_second_order() {
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

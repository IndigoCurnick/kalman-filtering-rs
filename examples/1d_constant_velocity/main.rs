use kalman_filtering_rs::{make_k, make_m, new_cov, write_to_file};
use peroxide::prelude::{eye, matrix, Matrix, Shape::Row};
use plotly::{common::Title, layout::Axis, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const SPEED: f64 = 2.0;
const TS: f64 = 0.1;
const DURATION: f64 = 60.0;
const SIGMA: f64 = 5.0;
const Q: f64 = 0.01;
const WRITE: bool = false;

fn main() {
    let data = get_data();
    let mut x = matrix(vec![3.0, 0.0], 2, 1, Row);
    let mut p = eye(2);

    let h = matrix(vec![1.0, 0.0], 1, 2, Row);
    let phi = phi();

    let mut position_history = vec![];
    let mut speed_history = vec![];

    let q = q();

    let mut x_residual = vec![];
    let mut x_m_residual = vec![];
    let mut v_residual = vec![];

    for i in 0..data.t.len() {
        let r = matrix(vec![SIGMA], 1, 1, Row);
        let x_star = data.x_m[i];
        let m = make_m(&phi, &p, &q);
        let k = make_k(&m, &h, &r);

        let xkminus1 = x.data[0];
        let xdotkminus1 = x.data[1];

        let xdot_bar = xdotkminus1;
        let x_bar = xkminus1 + 1.0 * xdotkminus1;

        let k1 = k[(0, 0)];
        let k2 = k[(1, 0)];

        let x_tilde = x_star - x_bar;

        let x_hat = x_bar + k1 * x_tilde;
        let xdot_hat = xdot_bar + k2 * x_tilde;

        position_history.push(x_hat);
        speed_history.push(xdot_hat);

        x_residual.push(data.x[i] - x_hat);
        x_m_residual.push(data.x[i] - x_star);
        v_residual.push(SPEED - xdot_hat);

        x = matrix(vec![x_hat, xdot_hat], 2, 1, Row);
        p = new_cov(&k, &h, &m);
    }

    // Distance
    let mut plot = Plot::new();
    let measurement_trace = Scatter::new(data.t.clone(), data.x_m.clone()).name("Measurement");
    let true_trace = Scatter::new(data.t.clone(), data.x.clone()).name("True");
    let filter_trace = Scatter::new(data.t.clone(), position_history).name("Filter");
    plot.add_traces(vec![measurement_trace, true_trace, filter_trace]);
    let layout = Layout::default()
        .title(Title::new("Position"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("Position (m)")));
    plot.set_layout(layout);
    let position_plot = plot.to_inline_html("position-plot");
    plot.show();

    let mut plot = Plot::new();

    // Speed
    let measurement_trace = Scatter::new(
        vec![data.t[0], data.t[data.t.len() - 1]],
        vec![SPEED, SPEED],
    )
    .name("True");
    let filter_trace = Scatter::new(data.t.clone(), speed_history).name("Filter");
    plot.add_traces(vec![measurement_trace, filter_trace]);
    let layout = Layout::default()
        .title(Title::new("Velocity"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("Velocity (m/s)")));
    plot.set_layout(layout);
    let velocity_plot = plot.to_inline_html("velocity-plot");
    plot.show();

    // Position Residual

    let mut plot = Plot::new();
    let trace = Scatter::new(data.t.clone(), x_residual).name("Filter to true residual");
    let trace2 = Scatter::new(data.t.clone(), x_m_residual).name("Measurement to true residual");
    plot.add_traces(vec![trace2, trace]);
    let layout = Layout::default()
        .title(Title::new("Position Residual"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("Position (m)")));
    plot.set_layout(layout);
    let position_residual = plot.to_inline_html("position-residual");
    plot.show();

    // Velocity residual
    let mut plot = Plot::new();
    let trace = Scatter::new(data.t.clone(), v_residual);
    plot.add_traces(vec![trace]);
    let layout = Layout::default()
        .title(Title::new("Velocity Residual"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("Velocity (m/s)")));
    plot.set_layout(layout);
    let velocity_residual = plot.to_inline_html("velocity-residual");
    plot.show();

    if WRITE {
        write_to_file("position-plot.html.tera", &position_plot);
        write_to_file("velocity-plot.html.tera", &velocity_plot);
        write_to_file("position-residual.html.tera", &position_residual);
        write_to_file("velocity-residual.html.tera", &velocity_residual);
    }
}

fn get_data() -> Data {
    let mut xs = vec![];
    let mut ts = vec![];
    let mut xms = vec![];
    let mut x = 0.0;
    let mut t = 0.0;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, SIGMA).unwrap();

    while t < DURATION {
        xs.push(x);
        ts.push(t);

        let r = normal.sample(&mut rng);

        xms.push(x + r);

        x += SPEED;
        t += TS;
    }

    return Data {
        t: ts,
        x: xs,
        x_m: xms,
    };
}

fn q() -> Matrix {
    let q = matrix(
        vec![
            TS.powf(3.0) / 3.0,
            TS.powf(2.0) / 2.0,
            TS.powf(2.0) / 2.0,
            TS,
        ],
        2,
        2,
        Row,
    );

    return Q * q;
}

fn phi() -> Matrix {
    return matrix(vec![1.0, TS, 0.0, 1.0], 2, 2, Row);
}

struct Data {
    pub t: Vec<f64>,
    pub x: Vec<f64>,
    pub x_m: Vec<f64>,
}

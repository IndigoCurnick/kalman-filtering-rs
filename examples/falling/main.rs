use std::{fs::File, io::Write};

use kalman_filtering_rs::{make_k, make_m, new_cov};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, layout::Axis, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const SIGNOISE: f64 = 300.0;
const PHIS: f64 = 0.001;
const TS: f64 = 0.1;
const INIT_S: f64 = 130_000.0;
const INIT_U: f64 = -2000.0;
const G: f64 = -9.81;
const MAXT: f64 = 60.0;
const WRITE: bool = false;

fn main() {
    let data = get_data();

    let mut state = matrix(vec![0.0, 0.0, 0.0], 3, 1, Row);

    let mut cov = zeros(3, 3);
    cov[(0, 0)] = 99999999.0;
    cov[(1, 1)] = 99999999.0;
    cov[(2, 2)] = 99999999.0;

    let h = matrix(vec![1.0, 0.0, 0.0], 1, 3, Row);
    let r = matrix(vec![SIGNOISE], 1, 1, Row);

    let mut x_history = vec![];
    let mut v_history = vec![];
    let mut a_history = vec![];

    let mut x_residual = vec![];
    let mut v_residual = vec![];
    let mut a_residual = vec![];

    let mut x_measurement_residual = vec![];

    for i in 0..data.t.len() {
        let x_star = data.x[i];

        let xkminus1 = state.data[0];
        let xdotkminus1 = state.data[1];
        let xdotdotkminus1 = state.data[2];

        let xdotdot_bar = xdotdotkminus1;
        let xdot_bar = xdotkminus1 + xdotdot_bar * TS;
        let x_bar = xkminus1 + xdot_bar * TS + 0.5 * xdotdot_bar * TS.powf(2.0);

        let phi = phi(TS);
        let q = q(TS);
        let m = make_m(&phi, &cov, &q);
        let k = make_k(&m, &h, &r);

        let x_tilda = x_star - x_bar;

        let k1 = k[(0, 0)];
        let k2 = k[(1, 0)];
        let k3 = k[(2, 0)];

        let x_hat = x_bar + k1 * x_tilda;
        let xdot_hat = xdot_bar + k2 * x_tilda;
        let xdotdot_hat = xdotdot_bar + k3 * x_tilda;

        state = matrix(vec![x_hat, xdot_hat, xdotdot_hat], 3, 1, Row);
        cov = new_cov(&k, &h, &m);

        x_history.push(x_hat);
        v_history.push(xdot_hat);
        a_history.push(xdotdot_hat);

        x_residual.push(data.s[i] - x_hat);
        v_residual.push(data.v[i] - xdot_hat);
        a_residual.push(G - xdotdot_hat);

        x_measurement_residual.push(data.x[i] - data.s[i]);
    }

    // Position Plot
    let mut plot = Plot::new();
    let m_trace = Scatter::new(data.t.clone(), data.s.clone()).name("Truth");
    let x_trace = Scatter::new(data.t.clone(), x_history).name("Filter");
    let s_trace = Scatter::new(data.t.clone(), data.x.clone()).name("Measurements");
    let layout = Layout::default()
        .title(Title::new("Position"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("Position (m)")));
    plot.set_layout(layout);
    plot.add_traces(vec![m_trace, x_trace, s_trace]);
    let position_plot = plot.to_inline_html("position-plot");
    plot.show();

    // Velocity plot
    let mut plot = Plot::new();
    let filter_trace = Scatter::new(data.t.clone(), v_history).name("Filter Velocity");
    let real_trace = Scatter::new(data.t.clone(), data.v.clone()).name("Real Velocity");
    let layout = Layout::default()
        .title(Title::new("Velocity"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("Velocity (m/s)")));
    plot.set_layout(layout);
    plot.add_traces(vec![filter_trace, real_trace]);
    let velocity_plot = plot.to_inline_html("velocity-plot");
    plot.show();

    // Acceleration plot
    let mut plot = Plot::new();
    let trace = Scatter::new(data.t.clone(), a_history).name("Filter Acceleration");
    let ideal_trace =
        Scatter::new(vec![data.t[0], data.t[data.t.len() - 1]], vec![G, G]).name("Ideal");
    plot.add_traces(vec![trace, ideal_trace]);
    let layout = Layout::default()
        .title(Title::new("Acceleration"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(
            Axis::default()
                .title(Title::new("Acceleration (m/s^2)"))
                .range(vec![-20.0, 20.0]),
        );
    plot.set_layout(layout);
    let acceleration_plot = plot.to_inline_html("acceleration-plot");
    plot.show();

    // Position residual
    let mut plot = Plot::new();
    let trace = Scatter::new(data.t.clone(), x_residual).name("Filter to true residual");
    let trace2 =
        Scatter::new(data.t.clone(), x_measurement_residual).name("Measurement to true residual");
    plot.add_traces(vec![trace, trace2]);
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
        .y_axis(
            Axis::default()
                .title(Title::new("Velocity (m/s)"))
                .range(vec![-50.0, 50.0]),
        );
    plot.set_layout(layout);
    let velocity_residual = plot.to_inline_html("velocity-residual");
    plot.show();

    // Acceleration residual
    let mut plot = Plot::new();
    let trace = Scatter::new(data.t.clone(), a_residual);
    plot.add_traces(vec![trace]);
    let layout = Layout::default()
        .title(Title::new("Acceleration Residual"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(
            Axis::default()
                .title(Title::new("Acceleration (m/s^2)"))
                .range(vec![-50.0, 50.0]),
        );
    plot.set_layout(layout);
    let acceleration_residual = plot.to_inline_html("acceleration-residual");
    plot.show();

    if WRITE {
        write_to_file("position-plot.html", &position_plot);
        write_to_file("velocity-plot.html", &velocity_plot);
        write_to_file("acceleration-plot.html", &acceleration_plot);
        write_to_file("position-residual.html", &position_residual);
        write_to_file("velocity-residual.html", &velocity_residual);
        write_to_file("acceleration-residual.html", &acceleration_residual);
    }
}

struct Data {
    pub s: Vec<f64>, // True value distance
    pub x: Vec<f64>, // Measurement distance
    pub t: Vec<f64>, // Time
    pub v: Vec<f64>, // velocity
}

fn get_data() -> Data {
    let mut s = INIT_S;
    let mut t = 0.0;
    // let mut u = ;
    let mut u = INIT_U;
    let g = G;
    let dt = TS;

    let mut s_history = vec![];
    let mut x_history = vec![];
    let mut t_history = vec![];
    let mut v_history = vec![];

    let normal = Normal::new(0.0, SIGNOISE).unwrap();
    let mut rng = rand::thread_rng();

    while t < MAXT {
        // Measurement

        s_history.push(s);
        t_history.push(t);
        v_history.push(u);
        x_history.push(s + normal.sample(&mut rng));

        // Propagate
        let v = u + g * dt;
        let d = 0.5 * (u + v) * dt;

        s += d;
        u = v;
        t += dt;
    }

    return Data {
        s: s_history,
        x: x_history,
        t: t_history,
        v: v_history,
    };
}

fn phi(dt: f64) -> Matrix {
    let phi = matrix(
        vec![
            1.0,
            dt,
            0.5 * dt.powf(2.0), //
            0.0,
            1.0,
            dt, //
            0.0,
            0.0,
            1.0, //
        ],
        3,
        3,
        Row,
    );

    return phi;
}

fn q(dt: f64) -> Matrix {
    let q = matrix(
        vec![
            dt.powf(5.0) / 20.0,
            dt.powf(4.0) / 8.0,
            dt.powf(3.0) / 6.0, //
            dt.powf(4.0) / 8.0,
            dt.powf(3.0) / 3.0,
            dt.powf(2.0) / 2.0, //
            dt.powf(3.0) / 6.0,
            dt.powf(2.0) / 2.0,
            dt,
        ],
        3,
        3,
        Row,
    );

    return PHIS * q;
}

fn write_to_file(file_name: &str, content: &String) {
    let mut file = File::create(file_name).expect("Failed to create file");
    file.write_all(content.as_bytes())
        .expect("Failed to write into the file");
}

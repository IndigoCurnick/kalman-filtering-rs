use kalman_filtering_rs::{make_k, make_m, new_cov, write_to_file};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, layout::Axis, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const SIGNOISE: f64 = 304.8;
const PHIS: f64 = 1.0; // In the book this value is given as 0, but this makes little sense to me
const G: f64 = 9.81;
const WRITE: bool = true;

fn main() {
    let measurements = get_data();

    let mut state = zeros(2, 1);
    let mut cov = zeros(2, 2);
    cov[(0, 0)] = 999999999.;
    cov[(1, 1)] = 999999999.;

    let h = matrix(vec![1.0, 0.0], 1, 2, Row);
    let r = matrix(vec![SIGNOISE], 1, 1, Row);

    let mut x_measurements = vec![];
    let mut x_truth = vec![];
    let mut v_truth = vec![];
    let mut x_history = vec![];
    let mut v_history = vec![];
    let mut t_history = vec![];
    let mut x_residual = vec![];
    let mut x_measurement_residual = vec![];
    let mut v_residual = vec![];

    let mut t = 0.0;
    for mea in &measurements {
        let dt = mea.t - t;

        let phi = phi(dt);
        let q = q(dt);

        let x_star = mea.x;

        let x_k_minus1 = state.data[0];
        let x_dot_k_minus1 = state.data[1];

        let x_tilde = x_star - x_k_minus1 - x_dot_k_minus1 * dt + 0.5 * G * dt.powf(2.0);

        let m = make_m(&phi, &cov, &q);
        let k = make_k(&m, &h, &r);

        let k1 = k[(0, 0)];
        let k2 = k[(1, 0)];

        let x_hat = x_k_minus1 + x_dot_k_minus1 * dt - 0.5 * G * dt.powf(2.0) + k1 * x_tilde;
        let x_dot_hat = x_dot_k_minus1 - G * dt + k2 * x_tilde;

        state = matrix(vec![x_hat, x_dot_hat], 1, 2, Row);
        cov = new_cov(&k, &h, &m);
        t = mea.t;

        x_history.push(state.data[0]);
        v_history.push(state.data[1]);
        x_measurements.push(mea.x);
        x_truth.push(mea.s);
        v_truth.push(mea.v);
        t_history.push(t);
        x_residual.push(x_hat - mea.s);
        x_measurement_residual.push(x_star - mea.s);
        v_residual.push(x_dot_hat - mea.v);
    }

    // Distance
    let mut x_plot = Plot::new();
    let m_trace = Scatter::new(t_history.clone(), x_measurements).name("Measurements");
    let x_trace = Scatter::new(t_history.clone(), x_history).name("Filter");
    let s_trace = Scatter::new(t_history.clone(), x_truth).name("Truth");
    x_plot.add_traces(vec![m_trace, x_trace, s_trace]);
    let layout = Layout::default()
        .title(Title::new("Position"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("Position (m)")));
    x_plot.set_layout(layout);
    x_plot.show();

    // Velocity
    let mut v_plot = Plot::new();
    let v_trace = Scatter::new(t_history.clone(), v_history).name("Filter Velocity");
    let v_truth = Scatter::new(t_history.clone(), v_truth).name("True Velocity");
    v_plot.add_traces(vec![v_trace, v_truth]);
    let layout = Layout::default()
        .title(Title::new("Velocity"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("Velocity (m/s)")));
    v_plot.set_layout(layout);
    v_plot.show();

    // Distance residual
    let mut xr_plot = Plot::new();
    let trace = Scatter::new(t_history.clone(), x_residual).name("Distance Residual");
    let m_trace =
        Scatter::new(t_history.clone(), x_measurement_residual).name("Measurement Residual");
    xr_plot.add_traces(vec![trace, m_trace]);
    let layout = Layout::default()
        .title(Title::new("Position Residual"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("Position (m)")));
    xr_plot.set_layout(layout);
    xr_plot.show();

    // velocity residual
    let mut vr_plot = Plot::new();
    let trace = Scatter::new(t_history.clone(), v_residual).name("Velocity Residual");
    vr_plot.add_trace(trace);
    let layout = Layout::default()
        .title(Title::new("Velocity Residual"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(
            Axis::default()
                .title(Title::new("Velocity (m/s)"))
                .range(vec![-50.0, 50.0]),
        );
    vr_plot.set_layout(layout);
    vr_plot.show();

    if WRITE {
        write_to_file(
            "position-plot.html.tera",
            &x_plot.to_inline_html("position-plot"),
        );
        write_to_file(
            "velocity-plot.html.tera",
            &v_plot.to_inline_html("velocity-plot"),
        );
        write_to_file(
            "position-residual.html.tera",
            &xr_plot.to_inline_html("position-residual"),
        );
        write_to_file(
            "velocity-residual.html.tera",
            &vr_plot.to_inline_html("velocity-plot"),
        );
    }
}

struct Measurement {
    pub s: f64, // True value
    pub x: f64, // Measurement
    pub t: f64, // Time
    pub v: f64,
}

fn get_data() -> Vec<Measurement> {
    let mut s = 121920.0;
    let mut t = 0.0;
    let mut u = -1828.8;
    let dt = 0.1;

    let mut m = vec![];

    let normal = Normal::new(0.0, SIGNOISE).unwrap();
    let mut rng = rand::thread_rng();

    while s > 0.0 {
        // Measurement
        let mes = Measurement {
            s: s,
            t: t,
            x: s + normal.sample(&mut rng),
            v: u,
        };
        m.push(mes);

        // Propagate
        let v = u - G * dt; // For now, all measurements 1s apart
        let d = 0.5 * (u + v) * dt; // Again, all measurements 1s apart

        s += d;
        u = v;
        t += dt;
    }

    return m;
}

fn phi(dt: f64) -> Matrix {
    let phi = matrix(vec![1.0, dt, 0.0, 1.0], 2, 2, Row);

    return phi;
}

fn q(dt: f64) -> Matrix {
    let mut q = zeros(2, 2);

    q[(0, 0)] = dt.powf(3.0) / 3.0;
    q[(0, 1)] = 0.5 * dt.powf(2.0);
    q[(1, 0)] = q[(0, 1)];
    q[(1, 1)] = dt;

    return PHIS * q;
}

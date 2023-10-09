use kalman_filtering_rs::{make_k, make_m, new_cov, write_to_file};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, layout::Axis, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const G: f64 = 9.81;
const INIT_ANGLE: f64 = 45.0; // Degrees
const INIT_VELOCITY: f64 = 915.0; //m/s
const TS: f64 = 0.1; // Time step s
const THETA_ERROR: f64 = 0.01; // radians
const R_ERROR: f64 = 20.5; // m
const RADAR_DIST: f64 = 30_500.0; // m
const EKFQ: f64 = 0.1;
const WRITE: bool = false;

fn main() {
    efk_sim();
}

fn efk_sim() {
    let data = get_data();

    let r_noise = matrix(
        vec![THETA_ERROR.powf(2.0), 0.0, 0.0, R_ERROR.powf(2.0)],
        2,
        2,
        Row,
    );
    let mut state = zeros(4, 1);
    state[(1, 0)] = data.vx[0];
    state[(3, 0)] = data.vy[0];

    println!("Opening state\n{}", state);
    let mut cov = zeros(4, 4);
    cov[(0, 0)] = 99999.9;
    cov[(1, 1)] = 99999.9;
    cov[(2, 2)] = 99999.9;
    cov[(3, 3)] = 99999.9;
    let q = q(EKFQ);

    let mut x_measurements = vec![];
    let mut y_measurements = vec![];

    let mut x_filter = vec![];
    let mut y_filter = vec![];

    let phi = matrix(
        vec![
            1.0, TS, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, TS, //
            0.0, 0.0, 0.0, 1.0, //
        ],
        4,
        4,
        Row,
    );

    let mut x_measurement_residual = vec![];
    let mut y_measurement_residual = vec![];
    let mut x_residual = vec![];
    let mut y_residual = vec![];

    let mut r_measurement_residual = vec![];
    let mut theta_measurement_residual = vec![];
    let mut r_residual = vec![];
    let mut theta_residual = vec![];

    for i in 0..data.r_measurements.len() {
        let theta_star = data.theta_measurements[i];
        let r_star = data.r_measurements[i];

        let x_star = x(r_star, theta_star);
        let y_star = y(r_star, theta_star);

        x_measurements.push(x_star);
        y_measurements.push(y_star);

        let xkminus1 = state.data[0];
        let xdotkminus1 = state.data[1];
        let ykminus1 = state.data[2];
        let ydotkminus1 = state.data[3];

        let x_bar = xkminus1 + TS * xdotkminus1;
        let xdot_bar = xdotkminus1;
        let y_bar = ykminus1 + TS * ydotkminus1 - 0.5 * G * TS.powf(2.0);
        let ydot_bar = ydotkminus1 - G * TS;

        let theta_bar = y_bar.atan2(x_bar - RADAR_DIST);
        // let theta_bar = (y_bar / x_bar - RADAR_DIST).atan();
        let r_bar = ((x_bar - RADAR_DIST).powf(2.0) + y_bar.powf(2.0)).sqrt();

        let h = matrix(
            vec![
                -y_bar / r_bar.powf(2.0),
                0.0,
                (x_bar - RADAR_DIST) / r_bar.powf(2.0),
                0.0,
                (x_bar - RADAR_DIST) / r_bar,
                0.0,
                y_bar / r_bar,
                0.0,
            ],
            2,
            4,
            Row,
        );

        let m = make_m(&phi, &cov, &q);
        let k = make_k(&m, &h, &r_noise);

        let k11 = k[(0, 0)];
        let k12 = k[(0, 1)];
        let k21 = k[(1, 0)];
        let k22 = k[(1, 1)];
        let k31 = k[(2, 0)];
        let k32 = k[(2, 1)];
        let k41 = k[(3, 0)];
        let k42 = k[(3, 1)];

        let theta_tilde = theta_star - theta_bar;
        let r_tilde = r_star - r_bar;

        let x_hat = x_bar + k11 * theta_tilde + k12 * r_tilde;
        let xdot_hat = xdot_bar + k21 * theta_tilde + k22 * r_tilde;
        let y_hat = y_bar + k31 * theta_tilde + k32 * r_tilde;
        let ydot_hat = ydot_bar + k41 * theta_tilde + k42 * r_tilde;

        x_filter.push(x_hat);
        y_filter.push(y_hat);

        state = matrix(vec![x_hat, xdot_hat, y_hat, ydot_hat], 4, 1, Row);
        cov = new_cov(&k, &h, &m);

        x_residual.push(x_hat - data.x[i]);
        y_residual.push(y_hat - data.y[i]);

        x_measurement_residual.push(x_star - data.x[i]);
        y_measurement_residual.push(y_star - data.y[i]);

        r_measurement_residual.push(r_star - data.r[i]);
        theta_measurement_residual.push(theta_star - data.theta[i]);

        let r_hat = r(x_hat, y_hat);
        let theta_hat = theta(x_hat, y_hat);

        r_residual.push(r_hat - data.r[i]);
        theta_residual.push(theta_hat - data.theta[i]);
    }

    // Plotting
    let mut full_plot = Plot::new();
    let ideal_trace = Scatter::new(data.x.clone(), data.y.clone()).name("Theory");
    full_plot.add_trace(ideal_trace);

    let measurement_trace = Scatter::new(x_measurements, y_measurements).name("Measurements");
    full_plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(x_filter, y_filter).name("Filter");
    full_plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("EKF"));
    full_plot.set_layout(layout);
    full_plot.show();

    // Plotting X Residuals
    let mut x_residual_plot = Plot::new();
    let trace =
        Scatter::new(data.time.clone(), x_measurement_residual).name("x measurements residuals");
    x_residual_plot.add_trace(trace);

    let trace = Scatter::new(data.time.clone(), x_residual).name("x residuals");
    x_residual_plot.add_trace(trace);
    let layout = Layout::default()
        .title(Title::new("x Residual - EKF"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("x (m)")));

    x_residual_plot.set_layout(layout);
    x_residual_plot.show();

    // Plotting Y residuals
    let mut y_residual_plot = Plot::new();
    let trace =
        Scatter::new(data.time.clone(), y_measurement_residual).name("y measurements residuals");
    y_residual_plot.add_trace(trace);

    let trace = Scatter::new(data.time.clone(), y_residual).name("y residuals");
    y_residual_plot.add_trace(trace);
    let layout = Layout::default()
        .title(Title::new("y Residual - EKF"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("y (m)")));

    y_residual_plot.set_layout(layout);
    y_residual_plot.show();

    // Plotting r Residuals
    let mut r_residual_plot = Plot::new();
    let trace =
        Scatter::new(data.time.clone(), r_measurement_residual).name("r measurements residuals");
    r_residual_plot.add_trace(trace);

    let trace = Scatter::new(data.time.clone(), r_residual).name("r residuals");
    r_residual_plot.add_trace(trace);
    let layout = Layout::default()
        .title(Title::new("r Residual - EKF"))
        .x_axis(Axis::default().title(Title::new("t (s)")))
        .y_axis(Axis::default().title(Title::new("r (m)")));

    r_residual_plot.set_layout(layout);
    r_residual_plot.show();

    // Plotting theta residuals
    let mut theta_residual_plot = Plot::new();
    let trace = Scatter::new(data.time.clone(), theta_measurement_residual)
        .name("theta measurements residuals");
    theta_residual_plot.add_trace(trace);

    let trace = Scatter::new(data.time.clone(), theta_residual).name("theta residuals");
    theta_residual_plot.add_trace(trace);
    let layout = Layout::default()
        .title(Title::new("theta Residual - EKF"))
        .x_axis(Axis::default().title(Title::new("theta (radians)")))
        .y_axis(
            Axis::default()
                .title(Title::new("y (m)"))
                .range(vec![-0.5, 0.5]),
        );

    theta_residual_plot.set_layout(layout);
    theta_residual_plot.show();

    if WRITE {
        write_to_file(
            "full-plot-ekf.html.tera",
            &full_plot.to_inline_html("full-plot-ekf"),
        );
        write_to_file(
            "x-residual-ekf.html.tera",
            &x_residual_plot.to_inline_html("x-residual-ekf"),
        );
        write_to_file(
            "y-residual-ekf.html.tera",
            &y_residual_plot.to_inline_html("y-residual-ekf"),
        );
        write_to_file(
            "r-residual-ekf.html.tera",
            &r_residual_plot.to_inline_html("r-residual-ekf"),
        );
        write_to_file(
            "theta-residual-ekf.html.tera",
            &theta_residual_plot.to_inline_html("theta-residual-ekf"),
        );
    }
}

fn get_data() -> Data {
    let mut t = 0.0;
    let mut x = 0.0;
    let mut y = 0.0;
    let vx = INIT_VELOCITY * INIT_ANGLE.to_radians().sin();
    let mut vy = INIT_VELOCITY * INIT_ANGLE.to_radians().cos();

    let mut t_history = vec![];
    let mut x_history = vec![];
    let mut y_history = vec![];
    let mut vx_history = vec![];
    let mut vy_history = vec![];
    let mut theta_measurements = vec![];
    let mut r_measurements = vec![];
    let mut r_ideal = vec![];
    let mut theta_ideal = vec![];

    let theta_normal = Normal::new(0.0, THETA_ERROR).unwrap();
    let r_normal = Normal::new(0.0, R_ERROR).unwrap();
    let mut rng = rand::thread_rng();

    while y >= 0.0 {
        t_history.push(t);
        x_history.push(x);
        y_history.push(y);
        vx_history.push(vx);
        vy_history.push(vy);

        let r = r(x, y);
        let theta = theta(x, y);

        theta_ideal.push(theta);
        r_ideal.push(r);
        theta_measurements.push(theta + theta_normal.sample(&mut rng));
        r_measurements.push(r + r_normal.sample(&mut rng));

        // y direction
        let tmp_vy = vy - G * TS;

        y = y + 0.5 * (vy + tmp_vy) * TS;

        vy = tmp_vy;

        // x direction
        x = x + vx * TS;

        // t
        t += TS;
    }

    return Data {
        time: t_history,
        x: x_history,
        vx: vx_history,
        y: y_history,
        vy: vy_history,
        r_measurements: r_measurements,
        theta_measurements: theta_measurements,
        r: r_ideal,
        theta: theta_ideal,
    };
}

fn theta(x: f64, y: f64) -> f64 {
    return y.atan2(x - RADAR_DIST);
    // return (y / (x - RADAR_DIST)).atan();
}

fn r(x: f64, y: f64) -> f64 {
    return ((x - RADAR_DIST).powf(2.0) + y.powf(2.0)).powf(0.5);
}

fn x(r: f64, theta: f64) -> f64 {
    return r * theta.cos() + RADAR_DIST;
}

fn y(r: f64, theta: f64) -> f64 {
    return r * theta.sin();
}

fn q(sf: f64) -> Matrix {
    let q = matrix(
        vec![
            TS.powf(3.0) / 3.0,
            TS.powf(2.0) / 2.0,
            0.0,
            0.0,
            TS.powf(2.0) / 2.0,
            TS,
            0.0,
            0.0,
            0.0,
            0.0,
            TS.powf(3.0) / 3.0,
            TS.powf(2.0) / 2.0,
            0.0,
            0.0,
            TS.powf(2.0) / 2.0,
            TS,
        ],
        4,
        4,
        Row,
    );

    return sf * q;
}

struct Data {
    pub time: Vec<f64>,
    pub x: Vec<f64>,
    pub vx: Vec<f64>,
    pub y: Vec<f64>,
    pub vy: Vec<f64>,
    pub r_measurements: Vec<f64>,
    pub theta_measurements: Vec<f64>,
    pub r: Vec<f64>,
    pub theta: Vec<f64>,
}

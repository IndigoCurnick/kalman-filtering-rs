use kalman_filtering_rs::{extended_predict, extended_update, predict, update};
use peroxide::{
    prelude::{diag, eye, matrix, zeros, Matrix, Shape::Row},
    statistics::stat,
};
use plotly::{common::Title, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const G: f64 = 9.81;
const INIT_ANGLE: f64 = 45.0; // Degrees
const INIT_VELOCITY: f64 = 915.0; //m/s
const TS: f64 = 0.1; // Time step s
const THETA_ERROR: f64 = 0.01; // radians
const R_ERROR: f64 = 20.5; // m
const RADAR_DIST: f64 = 30_500.0; // m
const Q: f64 = 1.0; // Q scaling factor

fn main() {
    linear_sim();
    efk_sim();
}

fn linear_sim() {
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
    let q = q();

    let mut x_measurements = vec![];
    let mut y_measurements = vec![];

    let g = matrix(vec![0.0, 0.0, -G * TS.powf(2.0) / 2.0, -G * TS], 4, 1, Row);
    let f = matrix(
        vec![
            1.0, TS, 0.0, 0.0, // These comments
            0.0, 1.0, 0.0, 0.0, // are just here
            0.0, 0.0, 1.0, TS, // to stop format
            0.0, 0.0, 0.0, 1.0, // putting this on one line
        ],
        4,
        4,
        Row,
    );

    let mut x_filter = vec![];
    let mut y_filter = vec![];

    let h = matrix(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 4, Row);

    for i in 0..data.r_measurements.len() {
        let theta_m = data.theta_measurements[i];
        let r_m = data.r_measurements[i];

        let x_m = x(r_m, theta_m);
        let y_m = y(r_m, theta_m);
        x_measurements.push(x_m);
        y_measurements.push(y_m);

        let z = matrix(vec![x_m, y_m], 2, 1, Row);

        let (new_state, new_cov) = predict(&state, &cov, &f, &q, Some(&g));
        let (new_state, new_cov) = update(&new_state, &new_cov, &h, &z, &r_noise);

        x_filter.push(new_state.data[0]);
        y_filter.push(new_state.data[2]);

        println!("state after\n{}", new_state);

        state = new_state;
        cov = new_cov;
    }

    // Plotting
    let mut plot = Plot::new();
    let ideal_trace = Scatter::new(data.x.clone(), data.y.clone()).name("Theory");
    plot.add_trace(ideal_trace);

    let measurement_trace = Scatter::new(x_measurements, y_measurements).name("Measurements");
    plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(x_filter, y_filter).name("Filter");
    plot.add_trace(filter_trace);

    let layout = Layout::default().title(Title::new("Fully Linear"));
    plot.set_layout(layout);
    plot.show();
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
    let q = q();

    let mut x_measurements = vec![];
    let mut y_measurements = vec![];

    let g = vec![0.0, 0.0, -G * TS.powf(2.0) / 2.0, -G * TS];
    let f = matrix(
        vec![
            1.0, TS, 0.0, 0.0, // These comments
            0.0, 1.0, 0.0, 0.0, // are just here
            0.0, 0.0, 1.0, TS, // to stop format
            0.0, 0.0, 0.0, 1.0, // putting this on one line
        ],
        4,
        4,
        Row,
    );

    let mut x_filter = vec![];
    let mut y_filter = vec![];

    for i in 0..data.r_measurements.len() {
        let theta_m = data.theta_measurements[i];
        let r_m = data.r_measurements[i];

        x_measurements.push(x(r_m, theta_m));
        y_measurements.push(y(r_m, theta_m));

        let z = matrix(vec![theta_m, r_m], 2, 1, Row);

        let current_x_estimate = state.data[0];
        let current_y_estimate = state.data[2];
        let current_r_estimate = r(current_x_estimate, current_y_estimate);
        // let current_theta_estimate = theta(current_x_estimate, current_y_estimate);
        let h = matrix(
            vec![
                -current_y_estimate / current_r_estimate.powf(2.0),
                0.0,
                (current_x_estimate - RADAR_DIST) / current_r_estimate.powf(2.0),
                0.0,
                (current_x_estimate - RADAR_DIST) / current_r_estimate,
                0.0,
                current_y_estimate / current_r_estimate,
                0.0,
            ],
            2,
            4,
            Row,
        );
        let (new_state, new_cov) = extended_predict(&f_func, &state, &cov, &f, &q, Some(&g));
        let (new_state, new_cov) = extended_update(&h_func, &new_state, &new_cov, &h, &z, &r_noise);

        x_filter.push(new_state.data[0]);
        y_filter.push(new_state.data[2]);

        println!("state after\n{}", new_state);

        state = new_state;
        cov = new_cov;
    }

    // Plotting
    let mut plot = Plot::new();
    let ideal_trace = Scatter::new(data.x.clone(), data.y.clone()).name("Theory");
    plot.add_trace(ideal_trace);

    let measurement_trace = Scatter::new(x_measurements, y_measurements).name("Measurements");
    plot.add_trace(measurement_trace);

    let filter_trace = Scatter::new(x_filter, y_filter).name("Filter");
    plot.add_trace(filter_trace);
    plot.show();
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

    let theta_normal = Normal::new(0.0, THETA_ERROR).unwrap();
    let r_normal = Normal::new(0.0, R_ERROR).unwrap();
    let mut rng = rand::thread_rng();

    while y >= 0.0 {
        t_history.push(t);
        x_history.push(x);
        y_history.push(y);
        vx_history.push(vx);
        vy_history.push(vy);

        theta_measurements.push(theta(x, y) + theta_normal.sample(&mut rng));
        r_measurements.push(r(x, y) + r_normal.sample(&mut rng));

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

fn f_func(x: &Vec<f64>, g: Option<&Vec<f64>>) -> Matrix {
    let f = matrix(
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

    let s = matrix(x.clone(), 4, 1, Row);

    let mut p = f * s;

    let k = g.unwrap();

    p[(2, 0)] += k[2];
    p[(3, 0)] += k[3];

    return p;
}

fn h_func(x: &Vec<f64>) -> Matrix {
    let current_x_estimate = x[0];
    let current_y_estimate = x[2];
    let current_r_estimate = r(current_x_estimate, current_y_estimate);
    let h = matrix(
        vec![
            -current_y_estimate / current_r_estimate.powf(2.0),
            0.0,
            (current_x_estimate - RADAR_DIST) / current_r_estimate.powf(2.0),
            0.0,
            (current_x_estimate - RADAR_DIST) / current_r_estimate,
            0.0,
            current_y_estimate / current_r_estimate,
            0.0,
        ],
        2,
        4,
        Row,
    );
    let x = matrix(x.clone(), 4, 1, Row);

    return h * x;
}

fn q() -> Matrix {
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

    return Q * q;
}

struct Data {
    pub time: Vec<f64>,
    pub x: Vec<f64>,
    pub vx: Vec<f64>,
    pub y: Vec<f64>,
    pub vy: Vec<f64>,
    pub r_measurements: Vec<f64>,
    pub theta_measurements: Vec<f64>,
}

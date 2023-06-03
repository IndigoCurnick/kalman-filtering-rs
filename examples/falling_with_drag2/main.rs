use peroxide::{
    fuga::LinearAlgebra,
    prelude::{eye, matrix, zeros, Matrix, Shape::Row},
};
use plotly::{common::Title, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const G: f64 = -9.81;
const TS: f64 = 0.1; // Seconds
const SIGNOISE: f64 = 304.8;
const INITX: f64 = 6705.0;
const PHI: f64 = 1.0;
const BETA: f64 = 100.0;

fn main() {
    let data = get_data();

    let h = matrix(vec![1.0, 0.0, 0.0], 1, 3, Row);
    let r = matrix(vec![SIGNOISE.powf(2.0)], 1, 1, Row);
    let mut state = zeros(3, 1);
    state[(2, 0)] = 1.0; // If beta = 0 then we get NaNs
    let mut cov = zeros(3, 3);
    cov[(0, 0)] = 99999.9;
    cov[(1, 1)] = 99999.9;
    cov[(2, 2)] = 99999.9;

    let mut x_history = vec![];
    let mut v_history = vec![];
    let mut beta_history = vec![];

    for i in 0..data.time.len() {
        let x_star = data.measured_positions_draggy[i];

        let psi = make_psi(state.data[0], state.data[1], state.data[2]);
        let q = q(state.data[0], state.data[1], state.data[2]);

        let m = &(&(&psi * &cov) * &psi.t()) + &q;
        let k = &(&m * &h.t()) * &(&(&(&h * &m) * &h.t()) + &r).inv();

        let k1 = k.data[0];
        let k2 = k.data[1];
        let k3 = k.data[2];

        let x_bar = state.data[0];
        let xdot_bar = state.data[1];
        let beta_bar = state.data[2];

        let x_tilde = x_star - x_bar;

        let x_hat = x_bar + k1 * x_tilde;
        let xdot_hat = xdot_bar + k2 * x_tilde;
        let beta_hat = beta_bar + k3 * x_tilde;

        println!("{}", state);

        x_history.push(x_hat);
        v_history.push(xdot_hat);
        beta_history.push(beta_hat);

        state = matrix(vec![x_hat, xdot_hat, beta_hat], 3, 1, Row);
        cov = &(eye(3) - &k * &h) * &m;
    }

    // Position
    let mut plot = Plot::new();

    let trace_drag = Scatter::new(data.time.clone(), data.draggy.position.clone()).name("Drag");
    let trace_nodrag =
        Scatter::new(data.time.clone(), data.no_drag.position.clone()).name("No Drag");
    let trace_measurements =
        Scatter::new(data.time.clone(), data.measured_positions_draggy.clone())
            .name("Measurements");
    let trace_filter = Scatter::new(data.time.clone(), x_history).name("filter");
    let layout = Layout::default().title(Title::new("Position"));
    plot.set_layout(layout);
    plot.add_traces(vec![
        trace_drag,
        trace_nodrag,
        trace_measurements,
        trace_filter,
    ]);
    plot.show();

    // Velocity
    let mut plot = Plot::new();

    let trace_drag = Scatter::new(data.time.clone(), data.draggy.velocity.clone()).name("Drag");
    let trace_nodrag =
        Scatter::new(data.time.clone(), data.no_drag.velocity.clone()).name("No Drag");
    let trace_history = Scatter::new(data.time.clone(), v_history).name("Filter");
    let layout = Layout::default().title(Title::new("Velocity"));

    plot.set_layout(layout);
    plot.add_traces(vec![trace_drag, trace_nodrag, trace_history]);
    plot.show();

    // Acceleration
    let mut plot = Plot::new();

    let trace_drag = Scatter::new(data.time.clone(), data.draggy.acceleration.clone()).name("Drag");
    let trace_nodrag =
        Scatter::new(data.time.clone(), data.no_drag.acceleration.clone()).name("No Drag");
    let layout = Layout::default().title(Title::new("Acceleration"));
    plot.set_layout(layout);
    plot.add_traces(vec![trace_drag, trace_nodrag]);
    plot.show();

    // Beta
    let mut plot = Plot::new();
    let layout = Layout::default().title(Title::new("Beta"));
    let ideal = Scatter::new(
        vec![data.time[0], data.time[data.time.len() - 1]],
        vec![BETA, BETA],
    )
    .name("Ideal");
    let beta_trace = Scatter::new(data.time.clone(), beta_history).name("Filter");
    plot.set_layout(layout);
    plot.add_traces(vec![ideal, beta_trace]);
    plot.show();
}

fn get_data() -> SimulationData {
    let mut t = 0.0;
    let mut x = INITX;
    let mut v = 0.0;
    let a = G;

    let mut x_drag = 6705.0;
    let mut v_drag = 0.0;
    let mut a_drag = G;

    let mut data = SimulationData::default();

    let normal = Normal::new(0.0, SIGNOISE).unwrap();
    let mut rng = rand::thread_rng();

    while x_drag > 0.0 {
        // No drag
        data.no_drag.position.push(x);
        data.no_drag.velocity.push(v);
        data.no_drag.acceleration.push(a);

        data.time.push(t);

        let u = v - G * TS;
        let d = 0.5 * (u + v) * TS;

        x -= d;
        v = u;
        t += TS;

        // Drag
        data.draggy.position.push(x_drag);
        data.draggy.velocity.push(v_drag);
        data.draggy.acceleration.push(a_drag);
        data.measured_positions_draggy
            .push(x_drag + normal.sample(&mut rng));

        let tmp_a = -calc_accel(x_drag, v_drag, BETA);

        let u = v_drag - tmp_a * TS;
        let d = 0.5 * (u + v_drag) * TS;

        x_drag -= d;
        v_drag = u;
        a_drag = tmp_a;
    }

    return data;
}

fn calc_accel(position: f64, velocity: f64, beta: f64) -> f64 {
    let q = 0.5 * rho(position) * velocity.powf(2.0);
    let a = (q * G) / beta - G;
    if a < 0.0 {
        return 0.0;
    } else {
        return a;
    }
}

#[derive(Default)]
struct SimulationData {
    pub draggy: SimulationUnit,
    pub no_drag: SimulationUnit,
    pub measured_positions_draggy: Vec<f64>,
    pub time: Vec<f64>,
}

#[derive(Default)]
struct SimulationUnit {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub acceleration: Vec<f64>,
}

fn make_psi(position: f64, velocity: f64, beta: f64) -> Matrix {
    let f21 = f21(position, velocity, beta);
    let f22 = f22(position, velocity, beta);
    let f23 = f23(position, velocity, beta);
    return matrix(
        vec![
            1.0,
            TS,
            0.0,
            f21 * TS,
            1.0 + f22 * TS,
            f23 * TS,
            0.0,
            0.0,
            1.0,
        ],
        3,
        3,
        Row,
    );
}

fn rho(position: f64) -> f64 {
    return 0.0034 * (-position / INITX).exp();
}

fn f21(position: f64, velocity: f64, beta: f64) -> f64 {
    let f = -(rho(position) * G * velocity.powf(2.0)) / (2.0 * INITX * beta);
    println!("f21 {}", f);
    return f;
}

fn f22(position: f64, velocity: f64, beta: f64) -> f64 {
    let f = (rho(position) * velocity * G) / beta;
    println!("f22 {}", f);
    return f;
}

fn f23(position: f64, velocity: f64, beta: f64) -> f64 {
    let f = (-rho(position) * G * velocity.powf(2.0)) / (2.0 * INITX * beta);
    println!("f23 {}", f);
    return f;
}

fn q(position: f64, velocity: f64, beta: f64) -> Matrix {
    let f23 = f23(position, velocity, beta);

    let q = matrix(
        vec![
            0.0,
            0.0,
            0.0,
            0.0,
            f23.powf(2.0) * TS.powf(3.0) / 3.0,
            f23 * TS.powf(2.0) / 2.0,
            0.0,
            f23 * TS.powf(2.0) / 2.0,
            TS,
        ],
        3,
        3,
        Row,
    );
    return PHI * q;
}

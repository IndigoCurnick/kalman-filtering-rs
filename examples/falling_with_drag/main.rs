use kalman_filtering_rs::{extended_predict, extended_update};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const G: f64 = -9.81;
const BETA: f64 = 100.0; // Book suggests values from 2,500-10,000kg/m^2
const TS: f64 = 0.1; // Seconds
const SIGNOISE: f64 = 304.8;
const INITX: f64 = 6705.0;
const PHI: f64 = 0.1;

fn main() {
    let data = get_data();

    let h = matrix(vec![1.0, 0.0], 1, 2, Row);
    let r = matrix(vec![SIGNOISE], 1, 1, Row);
    let mut state = zeros(2, 2);
    let mut cov = zeros(2, 2);
    cov[(0, 0)] = 99999.9;
    cov[(1, 1)] = 99999.9;

    let mut x_history = vec![];
    let mut v_history = vec![];
    for i in 0..data.time.len() {
        let z = matrix(vec![data.measured_positions_draggy[i]], 1, 1, Row);

        let f = make_f(state.data[0], state.data[1]);
        let q = q(state.data[0], state.data[1]);

        let (new_x, new_cov) = extended_predict(&f_func, &state, &cov, &f, &q, None);
        let (new_x, new_cov) = extended_update(&h_func, &new_x, &new_cov, &h, &z, &r, None);

        state = new_x;
        cov = new_cov;

        x_history.push(state.data[0]);
        v_history.push(state.data[1]);
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

        let tmp_a = -calc_accel(x_drag, v_drag);

        let u = v_drag - tmp_a * TS;
        let d = 0.5 * (u + v_drag) * TS;

        x_drag -= d;
        v_drag = u;
        a_drag = tmp_a;
    }

    return data;
}

fn calc_accel(position: f64, velocity: f64) -> f64 {
    let q = 0.5 * rho(position) * velocity.powf(2.0);
    let a = (q * G) / BETA - G;
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

fn make_f(position: f64, velocity: f64) -> Matrix {
    return matrix(
        vec![0.0, 1.0, f21(position, velocity), f22(position, velocity)],
        2,
        2,
        Row,
    );
}

fn rho(position: f64) -> f64 {
    return 0.0034 * (-position / INITX).exp();
}

fn f21(position: f64, velocity: f64) -> f64 {
    return -(rho(position) * G * velocity.powf(2.0)) / (2.0 * INITX * BETA);
}

fn f22(position: f64, velocity: f64) -> f64 {
    return (rho(position) * velocity * G) / BETA;
}

fn q(position: f64, velocity: f64) -> Matrix {
    let f22 = f22(position, velocity);
    let q = matrix(
        vec![
            TS.powf(3.0) / 3.0,
            TS.powf(2.0) / 2.0 + (f22 * TS.powf(3.0)) / 3.0,
            TS.powf(2.0) / 2.0 + (f22 * TS.powf(3.0)) / 3.0,
            TS + f22 + TS.powf(2.0) + f22.powf(2.0) * TS.powf(3.0) / 3.0,
        ],
        2,
        2,
        Row,
    );
    return PHI * q;
}

fn f_func(x: &Vec<f64>, _u: Option<&Vec<f64>>) -> Matrix {
    let s = x[0];
    let u = x[1];

    let ahat = -calc_accel(s, u);

    let v = u - ahat * TS;
    let d = 0.5 * (u + v) * TS;

    return matrix(vec![s - d, v], 2, 1, Row);
}

fn h_func(x: &Vec<f64>, _: Option<&Vec<f64>>) -> Matrix {
    let h = matrix(vec![1.0, 0.0], 1, 2, Row);
    let l = x.len();
    let x = matrix(x.clone(), l, 1, Row);
    return h * x;
}

use kalman_filtering_rs::{extended_predict, extended_update};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const G: f64 = -9.81;
const TS: f64 = 0.1; // Seconds
const SIGNOISE: f64 = 304.8;
const INITX: f64 = 6705.0;
const PHI: f64 = 1000.0;

fn main() {
    let data = get_data();

    let h = matrix(vec![1.0, 0.0, 0.0], 1, 3, Row);
    let r = matrix(vec![SIGNOISE], 1, 1, Row);
    let mut state = zeros(3, 1);
    state[(2, 0)] = 1.0; // If beta = 0 then we get NaNs
    let mut cov = zeros(3, 3);
    cov[(0, 0)] = 99999.9;
    cov[(1, 1)] = 99999.9;
    cov[(2, 2)] = 99999.9;

    let mut x_history = vec![];
    let mut v_history = vec![];
    for i in 0..data.time.len() {
        let z = matrix(vec![data.measured_positions_draggy[i]], 1, 1, Row);

        let f = make_f(state.data[0], state.data[1], state.data[2]);
        let q = q(state.data[0], state.data[1], state.data[2], TS);

        let (new_x, new_cov) = extended_predict(&f_func, &state, &cov, &f, &q, None);
        let (new_x, new_cov) = extended_update(&h_func, &new_x, &new_cov, &h, &z, &r, None);

        state = new_x;
        cov = new_cov;

        println!("{}", state);

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
    const BETA: f64 = 100.0;
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

fn make_f(position: f64, velocity: f64, beta: f64) -> Matrix {
    return matrix(
        vec![
            0.0,
            1.0,
            0.0,
            f21(position, velocity, beta),
            f22(position, velocity, beta),
            f23(position, velocity, beta),
            0.0,
            0.0,
            0.0,
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

fn q(position: f64, velocity: f64, beta: f64, dt: f64) -> Matrix {
    fn q11(f23: f64, dt: f64) -> f64 {
        return f23.powf(2.0) * (dt.powf(5.0) / 5.0);
    }

    fn q12(f22: f64, f23: f64, dt: f64) -> f64 {
        return f23.powf(2.0) + f22 * f23.powf(2.0) * (dt.powf(5.0) / 20.0);
    }

    fn q13(f23: f64, dt: f64) -> f64 {
        return f23 * (dt.powf(3.0) / 6.0);
    }

    fn q21(f22: f64, f23: f64, dt: f64) -> f64 {
        return f23.powf(2.0) * (dt.powf(4.0) / 8.0) + f22 * f23.powf(2.0) * (dt.powf(5.0) / 20.0);
    }

    fn q22(f22: f64, f23: f64, dt: f64) -> f64 {
        return f23.powf(2.0) * (dt.powf(3.0) / 6.0)
            + 2.0 * f22.powf(2.0) * f23.powf(2.0) * (dt.powf(5.0) / 20.0)
            + f22 * f23.powf(2.0) * (dt.powf(4.0) / 8.0);
    }

    fn q23(f22: f64, f23: f64, dt: f64) -> f64 {
        return f23 * (dt.powf(2.0) / 2.0) + f22 * f23 * (dt.powf(3.0) / 6.0);
    }

    fn q31(f23: f64, dt: f64) -> f64 {
        return f23 * (dt.powf(3.0) / 6.0);
    }

    fn q32(f22: f64, f23: f64, dt: f64) -> f64 {
        return f23 * (dt.powf(2.0) / 2.0) + f22 * f23 * (dt.powf(3.0) / 6.0);
    }

    fn q33(dt: f64) -> f64 {
        return dt;
    }

    let f21 = f21(position, velocity, beta);
    let f22 = f22(position, velocity, beta);
    let f23 = f23(position, velocity, beta);

    let q = matrix(
        vec![
            q11(f23, dt),
            q12(f22, f23, dt),
            q13(f23, dt),
            q21(f22, f23, dt),
            q22(f22, f23, dt),
            q23(f22, f23, dt),
            q31(f23, dt),
            q32(f22, f23, dt),
            q33(dt),
        ],
        3,
        3,
        Row,
    );
    return PHI * q;
}

fn f_func(x: &Vec<f64>, _u: Option<&Vec<f64>>) -> Matrix {
    let s = x[0];
    let u = x[1];
    let beta = x[2];
    println!("s {}, u {}, beta {}", s, u, beta);
    let ahat = -calc_accel(s, u, beta);
    println!("ahat {}", ahat);
    let v = u - ahat * TS;
    let d = 0.5 * (u + v) * TS;

    let f = matrix(vec![s - d, v, beta], 3, 1, Row);

    println!("f func {}", f);

    return f;
}

fn h_func(x: &Vec<f64>, _: Option<&Vec<f64>>) -> Matrix {
    let h = matrix(vec![1.0, 0.0, 0.0], 1, 3, Row);
    let l = x.len();
    let x = matrix(x.clone(), l, 1, Row);
    return h * x;
}

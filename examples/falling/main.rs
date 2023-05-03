use kalman_filtering_rs::{predict, update};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{Plot, Scatter};
use rand_distr::{Distribution, Normal};

const SIGNOISE: f64 = 304.8;
const PHIS: f64 = 0.1; // In the book this value is given as 0, but this makes little sense to me

fn main() {
    let measurements = get_data();

    let mut state = zeros(3, 1);
    let mut cov = zeros(3, 3);
    cov[(0, 0)] = 999999999.;
    cov[(1, 1)] = 999999999.;
    cov[(2, 2)] = 999999999.;

    let h = matrix(vec![1.0, 0.0, 0.0], 1, 3, Row);
    let r = matrix(vec![SIGNOISE], 1, 1, Row);

    let mut x_measurements = vec![];
    let mut truth = vec![];
    let mut x_history = vec![];
    let mut v_history = vec![];
    let mut a_history = vec![];
    let mut t = 0.0;
    for m in &measurements {
        let z = matrix(vec![m.x], 1, 1, Row);

        let dt = m.t - t;

        let f = f(dt);
        let q = q(dt);
        let (new_x, new_cov) = predict(&state, &cov, &f, &q);
        let (new_x, new_cov) = update(&new_x, &new_cov, &h, &z, &r);

        state = new_x;
        cov = new_cov;
        t = m.t;

        x_history.push(state.data[0]);
        v_history.push(state.data[1]);
        a_history.push(state.data[2]);
        x_measurements.push(m.x);
        truth.push(m.s);
    }

    let mut index = vec![];
    for i in 0..x_history.len() {
        index.push(i);
    }

    let mut plot = Plot::new();
    let m_trace = Scatter::new(index.clone(), x_measurements).name("Measurements");
    let x_trace = Scatter::new(index.clone(), x_history).name("Filter");
    let s_trace = Scatter::new(index.clone(), truth).name("Truth");
    plot.add_traces(vec![m_trace, x_trace, s_trace]);
    plot.show();

    let mut plot = Plot::new();
    let trace = Scatter::new(index.clone(), v_history).name("Filter Velocity");
    plot.add_trace(trace);
    plot.show();

    let mut plot = Plot::new();
    let trace = Scatter::new(index.clone(), a_history).name("Filter Acceleration");
    plot.add_trace(trace);
    plot.show();
}

struct Measurement {
    pub s: f64, // True value
    pub x: f64, // Measurement
    pub t: f64, // Time
}

fn get_data() -> Vec<Measurement> {
    let mut s = 121920.0;
    let mut t = 0.0;
    let mut u = 1828.8;
    let g = 9.8;
    let dt = 0.1;

    let mut m = vec![];

    let normal = Normal::new(0.0, SIGNOISE.powf(2.0)).unwrap();
    let mut rng = rand::thread_rng();

    while s > 0.0 {
        // Measurement
        let mes = Measurement {
            s: s,
            t: t,
            x: s + normal.sample(&mut rng),
        };
        m.push(mes);

        // Propagate
        let v = u + g * dt; // For now, all measurements 1s apart
        let d = 0.5 * (u + v) * dt; // Again, all measurements 1s apart

        s -= d;
        u = v;
        t += dt;
    }

    return m;
}

fn f(dt: f64) -> Matrix {
    let mut f = zeros(3, 3);
    f[(0, 0)] = 1.0;
    f[(0, 1)] = dt;
    f[(0, 2)] = 0.5 * dt.powf(2.0);
    f[(1, 1)] = 1.0;
    f[(1, 2)] = dt;
    f[(2, 2)] = 1.0;

    return f;
}

fn q(dt: f64) -> Matrix {
    let mut q = zeros(3, 3);

    q[(0, 0)] = dt.powf(5.0) / 20.0;
    q[(0, 1)] = dt.powf(4.0) / 8.0;
    q[(0, 2)] = dt.powf(3.0) / 6.0;
    q[(1, 0)] = q[(0, 1)];
    q[(1, 1)] = dt.powf(3.0) / 3.0;
    q[(1, 2)] = dt.powf(2.0) / 2.0;
    q[(2, 0)] = q[(0, 2)];
    q[(2, 1)] = q[(1, 2)];
    q[(2, 2)] = dt;

    return PHIS * q;
}

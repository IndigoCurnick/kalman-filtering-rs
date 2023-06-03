use kalman_filtering_rs::{extended_predict, extended_update};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const A: f64 = 1.0; // Amplitude
const OMEGA: f64 = 5.0; // Frequency
const TS: f64 = 0.01; // Time sampling
const R: f64 = 0.2;
const Q: f64 = 10.0;

fn main() {
    let data = get_data();

    let mut state = matrix(vec![1.0, 10.0, 3.0], 3, 1, Row);

    let mut cov = zeros(3, 3);
    cov[(0, 0)] = 999999.9;
    cov[(1, 1)] = 999999.9;
    cov[(2, 2)] = 999999.9;

    let mut y_filter: Vec<f64> = vec![];
    let mut a_filter = vec![];
    let mut omega_filter = vec![];
    let q = q();
    let r = matrix(vec![R.powf(2.0)], 1, 1, Row);
    let f = f();

    for i in 0..data.t.len() {
        let a = state.data[2];
        let phi = state.data[0];

        let h = matrix(vec![a * phi.cos(), 0.0, phi.sin()], 1, 3, Row);

        let z = matrix(vec![data.y_m[i]], 1, 1, Row);

        let (new_state, new_cov) = extended_predict(&f_func, &state, &cov, &f, &q, None);
        let (new_state, new_cov) = extended_update(&h_func, &new_state, &new_cov, &h, &z, &r, None);

        let estimated_y = new_state.data[2] * (new_state.data[1] * data.t[i]).sin();

        y_filter.push(estimated_y);
        a_filter.push(new_state.data[2]);
        omega_filter.push(new_state.data[1]);

        state = new_state;
        cov = new_cov;
    }
    // SIn wave
    let mut plot = Plot::new();
    let real = Scatter::new(data.t.clone(), data.y.clone()).name("Real");
    plot.add_trace(real);

    let measure = Scatter::new(data.t.clone(), data.y_m.clone()).name("Measurements");
    plot.add_trace(measure);

    let filter = Scatter::new(data.t.clone(), y_filter).name("Filter");
    plot.add_trace(filter);
    plot.show();

    // A
    let mut plot = Plot::new();
    let real =
        Scatter::new(vec![data.t[0], data.t.last().unwrap().clone()], vec![A, A]).name("Real");
    plot.add_trace(real);
    let filter = Scatter::new(data.t.clone(), a_filter).name("Filter");
    plot.add_trace(filter);
    let layout = Layout::default().title(Title::new("A"));
    plot.set_layout(layout);
    plot.show();

    // Omega
    let mut plot = Plot::new();
    let real = Scatter::new(
        vec![data.t[0], data.t.last().unwrap().clone()],
        vec![OMEGA, OMEGA],
    )
    .name("Real");
    plot.add_trace(real);
    let filter = Scatter::new(data.t.clone(), omega_filter).name("Filter");
    plot.add_trace(filter);
    let layout = Layout::default().title(Title::new("OMEGA"));
    plot.set_layout(layout);
    plot.show();
}

fn f() -> Matrix {
    return matrix(
        vec![
            1.0, TS, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
        ],
        3,
        3,
        Row,
    );
}

fn h_func(v: &Vec<f64>, _: Option<&Vec<f64>>) -> Matrix {
    let state = matrix(v.clone(), 3, 1, Row);
    let a = state.data[2];
    let phi = state.data[0];

    let h = matrix(vec![a * phi.cos(), 0.0, phi.sin()], 1, 3, Row);

    return h * state;
}

fn f_func(x: &Vec<f64>, _: Option<&Vec<f64>>) -> Matrix {
    let x = matrix(x.clone(), 3, 1, Row);
    let f = f();

    return f * x;
}

fn q() -> Matrix {
    let q = matrix(
        vec![
            TS.powf(3.0) / 3.0,
            TS.powf(2.0) / 2.0,
            0.0, //
            TS.powf(2.0) / 2.0,
            TS,
            0.0, //
            0.0,
            0.0,
            TS, //
        ],
        3,
        3,
        Row,
    );

    return Q * q;
}

fn get_data() -> Data {
    let mut t_hist = vec![];
    let mut y_hist = vec![];
    let mut y_m = vec![];
    let mut t = 0.0;

    let normal = Normal::new(0.0, R).unwrap();
    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let y = A * (OMEGA * t).sin();

        t_hist.push(t);
        y_hist.push(y);
        y_m.push(y + normal.sample(&mut rng));

        t += TS;
    }

    return Data {
        t: t_hist,
        y: y_hist,
        y_m: y_m,
    };
}

struct Data {
    pub t: Vec<f64>,
    pub y: Vec<f64>,
    pub y_m: Vec<f64>,
}

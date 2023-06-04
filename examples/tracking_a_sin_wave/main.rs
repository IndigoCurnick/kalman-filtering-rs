use kalman_filtering_rs::{make_k, make_m, new_cov};
use peroxide::prelude::{matrix, zeros, Matrix, Shape::Row};
use plotly::{common::Title, Layout, Plot, Scatter};
use rand_distr::{Distribution, Normal};

const A: f64 = 1.0; // Amplitude
const OMEGA: f64 = 5.0; // Frequency
const TS: f64 = 0.01; // Time sampling
const R: f64 = 0.2;
const Q: f64 = 1.0;

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
    let mut y_filter_alt = vec![];

    let q = q();
    let r = matrix(vec![R.powf(2.0)], 1, 1, Row);
    let phi = phi();

    for i in 0..data.t.len() {
        let phikminus1 = state.data[0];
        let omegakminus1 = state.data[1];
        let akminus1 = state.data[2];

        let x_star = data.y_m[i];

        let res = x_star - akminus1 * phikminus1.sin();

        let phi_bar = phikminus1 + omegakminus1 * TS;
        let omega_bar = omegakminus1;
        let a_bar = akminus1;

        let h = matrix(vec![a_bar * phi_bar.cos(), 0.0, phi_bar.sin()], 1, 3, Row);

        let m = make_m(&phi, &cov, &q);
        let k = make_k(&m, &h, &r);

        let k1 = k[(0, 0)];
        let k2 = k[(1, 0)];
        let k3 = k[(2, 0)];

        let phi_hat = phi_bar + k1 * res;
        let omega_hat = omega_bar + k2 * res;
        let a_hat = a_bar + k3 * res;

        let estimated_y = a_hat * (omega_hat * data.t[i]).sin();
        let alt_estimate_y = a_hat * phi_hat.sin();

        y_filter.push(estimated_y);
        y_filter_alt.push(alt_estimate_y);
        a_filter.push(a_hat);
        omega_filter.push(omega_hat);

        state = matrix(vec![phi_hat, omega_hat, a_hat], 3, 1, Row);
        cov = new_cov(&k, &h, &m);
    }
    // SIn wave
    let mut plot = Plot::new();
    let real = Scatter::new(data.t.clone(), data.y.clone()).name("Real");
    plot.add_trace(real);

    let measure = Scatter::new(data.t.clone(), data.y_m.clone()).name("Measurements");
    plot.add_trace(measure);

    let filter = Scatter::new(data.t.clone(), y_filter).name("Filter (omega * t)");
    plot.add_trace(filter);

    let filter = Scatter::new(data.t.clone(), y_filter_alt).name("Filter (phi)");
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

fn phi() -> Matrix {
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

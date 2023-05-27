use plotly::{Plot, Scatter};
use rand_distr::{Distribution, Normal};

const G: f64 = 9.81;
const INIT_ANGLE: f64 = 45.0; // Degrees
const INIT_VELOCITY: f64 = 915.0; //m/s
const TS: f64 = 0.1; // Time step s
const THETA_ERROR: f64 = 0.01; // radians
const R_ERROR: f64 = 20.5; // m
const RADAR_DIST: f64 = 30_500.0; // m

fn main() {
    let data = get_data();

    let mut plot = Plot::new();
    let ideal_trace = Scatter::new(data.x.clone(), data.y.clone()).name("Theory");
    plot.add_trace(ideal_trace);

    let mut x_measurements = vec![];
    let mut y_measurements = vec![];

    for i in 0..data.r_measurements.len() {
        x_measurements.push(x(data.r_measurements[i], data.theta_measurements[i]));
        y_measurements.push(y(data.r_measurements[i], data.theta_measurements[i]));
    }

    let measurement_trace = Scatter::new(x_measurements, y_measurements).name("Measurements");
    plot.add_trace(measurement_trace);
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

struct Data {
    pub time: Vec<f64>,
    pub x: Vec<f64>,
    pub vx: Vec<f64>,
    pub y: Vec<f64>,
    pub vy: Vec<f64>,
    pub r_measurements: Vec<f64>,
    pub theta_measurements: Vec<f64>,
}

use plotly::{Plot, Scatter};

const G: f64 = 9.81;
const INIT_ANGLE: f64 = 45.0; // Degrees
const INIT_VELOCITY: f64 = 915.0; //m/s
const TS: f64 = 0.1; // Time step s

fn main() {
    let data = get_data();

    let mut plot = Plot::new();
    let trace = Scatter::new(data.x.clone(), data.y.clone());
    plot.add_trace(trace);

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

    while y >= 0.0 {
        t_history.push(t);
        x_history.push(x);
        y_history.push(y);
        vx_history.push(vx);
        vy_history.push(vy);

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
    };
}

struct Data {
    pub time: Vec<f64>,
    pub x: Vec<f64>,
    pub vx: Vec<f64>,
    pub y: Vec<f64>,
    pub vy: Vec<f64>,
}

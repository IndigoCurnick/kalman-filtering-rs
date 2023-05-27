use plotly::{Plot, Scatter};
use rand_distr::{Distribution, Normal};

const TRUE_X: f64 = 5000.0;
const TRUE_Y: f64 = 300.0;
const R: f64 = 100.0;
const TS: f64 = 1.0;

fn main() {
    let data = get_data();

    let mut x_h = vec![];
    let mut y_h = vec![];

    let mut x_m = vec![];
    let mut y_m = vec![];

    for i in 0..data.r1.len() {
        let (x, y) = solve_position(
            data.r1[i],
            data.r2[i],
            data.xr1[i],
            data.yr1[i],
            data.xr2[i],
            data.yr2[i],
        );

        x_h.push(x);
        y_h.push(y);

        let (x, y) = solve_position(
            data.r1_m[i],
            data.r2_m[i],
            data.xr1[i],
            data.yr1[i],
            data.xr2[i],
            data.yr2[i],
        );

        x_m.push(x);
        y_m.push(x);
    }

    let mut plot = Plot::new();

    let ideal_ranges_trace = Scatter::new(x_h, y_h).name("Calculated from ideal ranges");
    plot.add_trace(ideal_ranges_trace);

    let true_trace = Scatter::new(vec![TRUE_X], vec![TRUE_Y]).name("True position");
    plot.add_trace(true_trace);

    let noisy_trace = Scatter::new(x_m, y_m).name("Calculated from noisy measurements");
    plot.add_trace(noisy_trace);

    plot.show();
}

fn solve_position(r1: f64, r2: f64, xr1: f64, yr1: f64, xr2: f64, yr2: f64) -> (f64, f64) {
    let alpha = -(yr2 - yr1) / (xr2 - xr1);
    let beta = (r1.powf(2.0) - r2.powf(2.0) - xr1.powf(2.0) - yr1.powf(2.0)
        + xr2.powf(2.0)
        + yr2.powf(2.0))
        / (2.0 * (xr2 - xr1));

    let a = 1.0 + alpha.powf(2.0);
    let b = -2.0 * alpha * xr1 + 2.0 * alpha * beta - 2.0 * yr1;
    let c = xr1.powf(2.0) - 2.0 * xr1 * beta + yr1.powf(2.0) - r1.powf(2.0);

    let y = (-b - (b.powf(2.0) - 4.0 * a * c).sqrt()) / (2.0 * a);
    let x = alpha * y + beta;

    return (x, -y);
}

fn get_data() -> Data {
    let mut x1 = -10_000.0;
    let mut y1 = 25_000.0;
    let mut vx1 = 10.0;

    let mut x2 = 10_000.0;
    let mut y2 = 20_000.0;
    let mut vx2 = -10.0;

    let mut r1_h = vec![];
    let mut r2_h = vec![];

    let mut r1_m = vec![];
    let mut r2_m = vec![];

    let mut xr1_h = vec![];
    let mut yr1_h = vec![];
    let mut xr2_h = vec![];
    let mut yr2_h = vec![];

    let normal = Normal::new(0.0, R).unwrap();
    let mut rng = rand::thread_rng();

    for i in 0..100 {
        let r1 = ((x1 - TRUE_X).powf(2.0) + (y1 - TRUE_Y).powf(2.0)).sqrt();
        let r2 = ((x2 - TRUE_X).powf(2.0) + (y2 - TRUE_Y).powf(2.0)).sqrt();

        r1_h.push(r1);
        r2_h.push(r2);

        r1_m.push(r1 + normal.sample(&mut rng));
        r2_m.push(r2 + normal.sample(&mut rng));

        xr1_h.push(x1);
        yr1_h.push(y1);

        xr2_h.push(x2);
        yr2_h.push(y2);

        x1 += vx1 * TS;
        x2 += vx2 * TS;
    }

    return Data {
        r1: r1_h,
        r2: r2_h,
        r1_m: r1_m,
        r2_m: r2_m,
        xr1: xr1_h,
        yr1: yr1_h,
        xr2: xr2_h,
        yr2: yr2_h,
    };
}

struct Data {
    pub r1: Vec<f64>,
    pub r2: Vec<f64>,
    pub r1_m: Vec<f64>,
    pub r2_m: Vec<f64>,
    pub xr1: Vec<f64>,
    pub yr1: Vec<f64>,
    pub xr2: Vec<f64>,
    pub yr2: Vec<f64>,
}

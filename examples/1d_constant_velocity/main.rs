use kalman_filtering_rs::{predict, update};
use peroxide::prelude::{eye, matrix, Shape::Row};
use plotly::{Plot, Scatter};
use rand_distr::{Distribution, Normal};

fn main() {
    let speed = 2.0;
    let measurements = get_data(0.0, speed, 30, 3.0);

    let mut index = vec![];

    let mut x = matrix(vec![3.0, 0.0], 2, 1, Row);
    let mut p = eye(2);

    let h = matrix(vec![1.0, 0.0], 1, 2, Row);
    let f = matrix(vec![1.0, 0.5, 0.0, 1.0], 2, 2, Row);

    let mut position_history = vec![];
    let mut speed_history = vec![];

    let q = matrix(vec![1.0 / 4.0, 1.0 / 2.0, 1.0 / 2.0, 1.0], 2, 2, Row);

    for m in &measurements {
        let z = matrix(vec![*m], 1, 1, Row);

        let r = matrix(vec![2.0], 1, 1, Row);

        let (new_x, new_p) = predict(&x, &p, &f, &q);
        let (new_x, new_p) = update(&new_x, &new_p, &h, &z, &r);

        position_history.push(new_x.data[0]);
        speed_history.push(new_x.data[1]);

        x = new_x;
        p = new_p;
    }

    let mut true_speeds = vec![];
    for i in 0..measurements.len() {
        index.push(i);
        true_speeds.push(speed);
    }

    let mut plot = Plot::new();

    let measurement_trace = Scatter::new(index.clone(), measurements).name("Measurement");
    let filter_trace = Scatter::new(index.clone(), position_history).name("Filter");
    plot.add_trace(measurement_trace);
    plot.add_trace(filter_trace);
    plot.show();

    let mut plot = Plot::new();

    let measurement_trace = Scatter::new(index.clone(), true_speeds).name("Speed");
    let filter_trace = Scatter::new(index.clone(), speed_history).name("Filter");
    plot.add_trace(measurement_trace);
    plot.add_trace(filter_trace);
    plot.show();
}

fn get_data(starting_point: f64, velocity: f64, n: usize, s: f64) -> Vec<f64> {
    let mut out = vec![];
    let mut x = starting_point;
    out.push(x);

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, s).unwrap();

    for _i in 0..n {
        x = x + velocity;

        let r = normal.sample(&mut rng);
        out.push(x + r);
    }

    return out;
}

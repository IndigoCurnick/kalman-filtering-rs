use peroxide::prelude::{eye, Matrix, SimplerLinearAlgebra};

pub fn update(x: &Matrix, p: &Matrix, h: &Matrix, z: &Matrix, r: &Matrix) -> (Matrix, Matrix) {
    let residual = z - &(h * x);
    let pht = p * &h.t();
    let s = &(h * &pht) + r;
    let s_inv = s.inv();
    let k = &pht * &s_inv;
    let new_x = x + &(&k * &residual);
    let i_kh = eye(x.row) - (&k * h);
    let new_cov = &(&(&i_kh * p) * &i_kh.t()) + &(&(&k * r) * &k.t());

    return (new_x, new_cov);
}

pub fn predict(
    x: &Matrix,
    p: &Matrix,
    f: &Matrix,
    q: &Matrix,
    g: Option<&Matrix>,
) -> (Matrix, Matrix) {
    let mut new_x = f * x;
    if g.is_some() {
        new_x = &new_x + g.unwrap();
    }
    let new_cov = &(&(f * p) * f) + q;

    return (new_x, new_cov);
}

pub fn extended_update(
    h_func: &dyn Fn(&Vec<f64>, Option<&Vec<f64>>) -> Matrix,
    x: &Matrix,
    p: &Matrix,
    h: &Matrix,
    z: &Matrix,
    r: &Matrix,
    o: Option<&Vec<f64>>,
) -> (Matrix, Matrix) {
    let residual = z - &h_func(&x.data, o);
    let pht = p * &h.t();
    let s = &(h * &pht) + r;
    let s_inv = s.inv();
    let k = &pht * &s_inv;
    let new_x = x + &(&k * &residual);
    let i_kh = eye(x.row) - (&k * h);
    let new_cov = &(&(&i_kh * p) * &i_kh.t()) + &(&(&k * r) * &k.t());

    return (new_x, new_cov);
}

pub fn extended_predict(
    f_func: &dyn Fn(&Vec<f64>, Option<&Vec<f64>>) -> Matrix,
    x: &Matrix,
    p: &Matrix,
    f: &Matrix,
    q: &Matrix,
    u: Option<&Vec<f64>>,
) -> (Matrix, Matrix) {
    let new_x = f_func(&x.data, u);

    let new_p = &(f * p * f.t()) + q;

    return (new_x, new_p);
}

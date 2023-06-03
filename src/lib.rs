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

pub fn make_m(phi: &Matrix, p: &Matrix, q: &Matrix) -> Matrix {
    return &(&(phi * p) * &phi.t()) + q;
}

pub fn make_k(m: &Matrix, h: &Matrix, r: &Matrix) -> Matrix {
    return (m * &h.t()) * (&(h * m * h.t()) + r).inv();
}

pub fn new_cov(k: &Matrix, h: &Matrix, m: &Matrix) -> Matrix {
    return &(eye(k.data.len()) - k * h) * m;
}

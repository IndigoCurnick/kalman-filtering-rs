use peroxide::prelude::{eye, Matrix, SimplerLinearAlgebra};

pub fn update(x: &Matrix, p: &Matrix, h: &Matrix, z: &Matrix, r: &Matrix) -> (Matrix, Matrix) {
    let residual = z - &(h * x);
    let pht = p * &h.t();
    let s = &(h * &pht) + r;
    let s_inv = s.inv();
    let k = &pht * &s_inv;
    let new_x = x + &(&k * &residual);
    let i_kh = eye(x.col) - (&k * h);
    let new_cov = &(&(&i_kh * p) * &i_kh.t()) + &(&(&k * r) * &k.t());

    return (new_x, new_cov);
}

pub fn predict(x: &Matrix, p: &Matrix, f: &Matrix, q: &Matrix) -> (Matrix, Matrix) {
    let new_x = f * x;
    let new_cov = &(&(f * p) * f) + q;

    return (new_x, new_cov);
}

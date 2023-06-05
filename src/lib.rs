use peroxide::{
    fuga::LinearAlgebra,
    prelude::{eye, Matrix},
};

pub fn update(x: &Matrix, p: &Matrix, h: &Matrix, z: &Matrix, r: &Matrix) -> (Matrix, Matrix) {
    // k
    let s = &(h * p * h.t()) + r;
    let s_inv = s.inv();
    let k = (p * &h.t()) * s_inv;

    // state
    let residual = z - &(h * x);
    println!("y {}", residual);
    let new_x = x + &(&k * &residual);

    // cov
    let t = &k * h;
    let i_kh = eye(t.col) - t;
    println!("ikh\n{}", i_kh);
    let new_cov = &(&(&i_kh * p) * &i_kh.t()) + &(&(&k * r) * &k.t());
    // let new_cov = 0.5 * &new_cov + 0.5 * &new_cov.t();
    // let new_cov = &i_kh * p;
    println!("new cov\n{}", new_cov);
    return (new_x, new_cov);
}

pub fn predict(
    x: &Matrix,
    p: &Matrix,
    phi: &Matrix,
    q: &Matrix,
    g: Option<&Matrix>,
) -> (Matrix, Matrix) {
    let mut new_x = phi * x;
    if g.is_some() {
        new_x = &new_x + g.unwrap();
    }
    let new_cov = &(&(phi * p) * phi) + q;

    return (new_x, new_cov);
}

pub fn make_m(phi: &Matrix, p: &Matrix, q: &Matrix) -> Matrix {
    return &(&(phi * p) * &phi.t()) + q;
}

pub fn make_k(m: &Matrix, h: &Matrix, r: &Matrix) -> Matrix {
    println!("m\n{}", m);
    println!("h\n{}", h);
    println!("r\n{}", r);
    return (m * &h.t()) * (&(h * m * h.t()) + r).inv();
}

pub fn new_cov(k: &Matrix, h: &Matrix, m: &Matrix) -> Matrix {
    let s = k * h;
    return &(eye(s.col) - s) * m;
}

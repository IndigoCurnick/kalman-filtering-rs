use std::{fs::File, io::Write};

use peroxide::{
    fuga::LinearAlgebra,
    prelude::{eye, Matrix},
};

pub fn make_m(phi: &Matrix, p: &Matrix, q: &Matrix) -> Matrix {
    return &(&(phi * p) * &phi.t()) + q;
}

pub fn make_k(m: &Matrix, h: &Matrix, r: &Matrix) -> Matrix {
    return (m * &h.t()) * (&(h * m * h.t()) + r).inv();
}

pub fn new_cov(k: &Matrix, h: &Matrix, m: &Matrix) -> Matrix {
    let s = k * h;
    return &(eye(s.col) - s) * m;
}

pub fn write_to_file(file_name: &str, content: &String) {
    let mut file = File::create(file_name).expect("Failed to create file");
    file.write_all(content.as_bytes())
        .expect("Failed to write into the file");
}

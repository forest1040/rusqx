use crate::{carray, carray_i, Qubit};
use ndarray::prelude::*;
use num::complex::Complex;
use once_cell::sync::Lazy;

#[derive(Debug)]
pub struct DoubleGate {
    matrix: Array2<Complex<f64>>,
}

macro_rules! gen_gates {
    ($mat: ident) => {
        #[allow(non_snake_case)]
        fn $mat(&mut self, qubit1: &Qubit, qubit2: &Qubit) {
            self.apply_double(&$mat.matrix, qubit1, qubit2);
        }
    };

    ($($ms: ident),*) => {
        $(gen_gates!($ms);)*
    };
}

pub trait DoubleGateApplicator {
    ///
    /// An operation for the given unitary matrix `matrix` to `qubit1` and `qubit2`
    ///
    fn apply_double(&mut self, matrix: &Array2<Complex<f64>>, qubit1: &Qubit, qubit2: &Qubit);

    gen_gates!(CNOT, SWAP, SQSWAP);

    // fn cphase(&mut self, phi: f64, qubit1: &Qubit, qubit2: &Qubit) {
    //     let mut matrix = carray![
    //         [1., 0., 0., 0.],
    //         [0., 1., 0., 0.],
    //         [0., 0., 1., 0.],
    //         [0., 0., 0., 0.]
    //     ];
    //     matrix[[3, 3]] = Complex::new(phi.cos(), phi.sin());
    //     self.apply_double(&matrix, qubit1, qubit2);
    // }
}

pub static CNOT: Lazy<DoubleGate> = {
    Lazy::new(|| DoubleGate {
        matrix: carray![
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.]
        ],
    })
};
pub static SWAP: Lazy<DoubleGate> = {
    Lazy::new(|| DoubleGate {
        matrix: carray![
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.]
        ],
    })
};
pub static SQSWAP: Lazy<DoubleGate> = {
    Lazy::new(|| DoubleGate {
        matrix: carray![
            [1., 0., 0., 0.],
            [0., 0.5, 0.5, 0.],
            [0., 0.5, 0.5, 0.],
            [0., 0., 0., 1.]
        ] + carray_i![
            [0., 0., 0., 0.],
            [0., 0.5, -0.5, 0.],
            [0., -0.5, 0.5, 0.],
            [0., 0., 0., 0.]
        ],
    })
};

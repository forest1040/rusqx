use crate::{carray, carray_i, Qubit};
use ndarray::prelude::*;
use num::complex::Complex;
use once_cell::sync::Lazy;

#[derive(Debug)]
pub struct SingleGate {
    pub matrix: Array2<Complex<f64>>,
}

macro_rules! gen_gates {
    ($mat: ident) => {
        #[allow(non_snake_case)]
        fn $mat(&mut self, qubit: &Qubit) {
            self.apply_single(&$mat.matrix, qubit);
        }
    };

    ($($ms: ident),*) => {
        $(gen_gates!($ms);)*
    };
}

pub trait SingleGateApplicator {
    ///
    /// An operation for the given unitary matrix `matrix` to `qubit`
    ///
    fn apply_single(&mut self, matrix: &Array2<Complex<f64>>, qubit: &Qubit);

    gen_gates!(H, X, Y, Z, ID);

    fn phase(&mut self, phi: f64, qubit: &Qubit) {
        let mut matrix = carray![[1., 0.], [0., 0.]];
        matrix[[1, 1]] = Complex::new(phi.cos(), phi.sin());
        self.apply_single(&matrix, qubit);
    }
}

pub static H: Lazy<SingleGate> = {
    Lazy::new(|| SingleGate {
        matrix: carray![[1., 1.], [1., -1.]] / (2f64).sqrt(),
    })
};
pub static X: Lazy<SingleGate> = {
    Lazy::new(|| SingleGate {
        matrix: carray![[0., 1.], [1., 0.]],
    })
};
pub static Y: Lazy<SingleGate> = {
    Lazy::new(|| SingleGate {
        matrix: carray_i![[0., 1.], [-1., 0.]],
    })
};
pub static Z: Lazy<SingleGate> = {
    Lazy::new(|| SingleGate {
        matrix: carray![[1., 0.], [0., -1.]],
    })
};
pub static ID: Lazy<SingleGate> = {
    Lazy::new(|| SingleGate {
        matrix: carray![[1., 0.], [0., 1.]],
    })
};
pub static SQNOT: Lazy<SingleGate> = {
    Lazy::new(|| SingleGate {
        matrix: carray![[1., 1.], [1., 1.]] / 2. + carray_i![[1., -1.], [-1., 1.]] / 2.,
    })
};

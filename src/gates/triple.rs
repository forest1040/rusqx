use crate::Qubit;
use ndarray::prelude::*;
use num::complex::Complex;
use once_cell::sync::Lazy;

#[derive(Debug)]
pub struct TripleGate {
    matrix: Array2<Complex<f64>>,
}

macro_rules! gen_gates {
    ($mat: ident) => {
        #[allow(non_snake_case)]
        fn $mat(&mut self, qubit1: &Qubit, qubit2: &Qubit, qubit3: &Qubit) {
            self.apply_triple(&$mat.matrix, qubit1, qubit2, qubit3);
        }
    };

    ($($ms: ident),*) => {
        $(gen_gates!($ms);)*
    };
}

macro_rules! carray {
    ( $([$($x: expr),*]),* ) => {{
        use num::complex::Complex;
        array![
            $([$(Complex::new($x, 0.)),*]),*
        ]
    }};
}

///
/// An trait for the types which accept operations for theree qubits.
///
pub trait TripleGateApplicator {
    ///
    /// An operation for the given unitary matrix `matrix` to `qubit1`, `qubit2` and `qubit3`
    ///
    fn apply_triple(
        &mut self,
        matrix: &Array2<Complex<f64>>,
        qubit1: &Qubit,
        qubit2: &Qubit,
        qubit3: &Qubit,
    );

    gen_gates!(CCNOT, CSWAP);
}

pub static CCNOT: Lazy<TripleGate> = {
    Lazy::new(|| TripleGate {
        matrix: carray![
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 1., 0.]
        ],
    })
};
pub static CSWAP: Lazy<TripleGate> = {
    Lazy::new(|| TripleGate {
        matrix: carray![
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.]
        ],
    })
};

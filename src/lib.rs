extern crate ndarray;
extern crate num;
extern crate rand;

pub mod gates;
pub mod prelude;
pub mod simulator;

#[macro_export]
macro_rules! carray {
    ( $([$($x: expr),*]),* ) => {{
        use num::complex::Complex;
        array![
            $([$(Complex::new($x, 0.)),*]),*
        ]
    }};
}

#[macro_export]
macro_rules! carray_i {
    ( $([$($x: expr),*]),* ) => {{
        use num::complex::Complex;
        array![
            $([$(Complex::new(0., $x)),*]),*
        ]
    }};
}

///
/// A type for the result of the measurement of a qubit.
///
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MeasuredResult {
    /// The qubit is measured as $|0\rangle$
    Zero,
    /// The qubit is measured as $|1\rangle$
    One,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Qubit {
    /// The index in a given quantum machine.
    pub index: usize,
}

pub trait QuantumMachine {
    /// Returns all the qubits in the machine.
    fn get_qubits(&self) -> Vec<Qubit>;

    /// Measures the given qubit.
    /// Note that the qubit is expected to be projected to the corresponding state.
    fn measure(&mut self, qubit: &Qubit) -> MeasuredResult;
}

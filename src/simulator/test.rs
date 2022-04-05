// use crate::gates::double::DoubleGateApplicator;
// use crate::gates::single::SingleGateApplicator;
// use crate::gates::triple::TripleGateApplicator;
use crate::{carray, carray_i, MeasuredResult, QuantumMachine, Qubit};
// use ndarray::prelude::*;
// use num::complex::Complex;
// use rand::{self, Rng};
use std::cmp;

#[cfg(test)]
mod tests {
    use super::*;
    use Qubit;

    #[test]
    fn test_single_gate() {
        //let qubit = &Qubit { index: 0 };
        let qubit = &Qubit { index: 2 };
        let mask = 1usize << qubit.index;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        let dim = 16;
        println!("mask     : {:0>64b}", mask);
        println!("mask_low : {:0>64b}", mask_low);
        println!("mask_high: {:0>64b}", mask_high);
        for state_index in 0..dim >> 1 {
            println!("state_index        : {:0>4b}", state_index);
            println!(
                "(state_index & mask_low): {:0>4b}",
                (state_index & mask_low)
            );
            println!(
                "((state_index & mask_high) << 1): {:0>4b}",
                ((state_index & mask_high) << 1)
            );
            let basis_0 = (state_index & mask_low) + ((state_index & mask_high) << 1);
            let basis_1 = basis_0 + mask;
            println!("{} {}", basis_0, basis_1);
        }
    }

    #[test]
    fn test_double_gate() {
        let qubit1 = &Qubit { index: 1 };
        let qubit2 = &Qubit { index: 2 };
        let qubits = &[qubit1, qubit2];
        let qubits_size = qubits.len();
        let min_qubit_index = cmp::min(qubit1.index, qubit2.index);
        let max_qubit_index = cmp::max(qubit1.index, qubit2.index);
        let min_qubit_mask = 1usize << min_qubit_index;
        let max_qubit_mask = 1usize << (max_qubit_index - 1);
        let low_mask = min_qubit_mask - 1;
        let mid_mask = (max_qubit_mask - 1) ^ low_mask;
        let high_mask = !(max_qubit_mask - 1);
        // let target_mask1 = 1 << qubit1.index;
        // let target_mask2 = 1 << qubit2.index;
        let target_mask1 = 1 << qubit2.index;
        let target_mask2 = 1 << qubit1.index;
        // loop variables
        let dim = 16;
        for state_index in 0..dim >> qubits_size {
            // create index
            let basis_0 = (state_index & low_mask)
                + ((state_index & mid_mask) << 1)
                + ((state_index & high_mask) << 2);
            // gather index
            let basis_1 = basis_0 + target_mask1;
            let basis_2 = basis_0 + target_mask2;
            let basis_3 = basis_1 + target_mask2;
            println!("{} {} {} {}", basis_0, basis_1, basis_2, basis_3);
        }
    }
}

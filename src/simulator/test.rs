use crate::Qubit;
use std::cmp;

#[cfg(test)]
mod tests {
    use crate::simulator::simulator::{indices_vec2, mask_vec2};

    use super::*;
    use Qubit;

    #[test]
    fn test_single_gate() {
        //let qubit = &Qubit { index: 0 };
        let qubit = &Qubit { index: 1 };
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
        let qubit1 = &Qubit { index: 0 };
        let qubit2 = &Qubit { index: 1 };
        let qubits = &[qubit1, qubit2];
        let qubits_size = qubits.len();
        let min_qubit_index = cmp::min(qubit1.index, qubit2.index);
        let max_qubit_index = cmp::max(qubit1.index, qubit2.index);
        let min_qubit_mask = 1usize << min_qubit_index;
        let max_qubit_mask = 1usize << (max_qubit_index - 1);
        // 間の部分（例えば2bitゲートでindexが0と2の場合、間のゲートがないindex1の部分）
        let low_mask = min_qubit_mask - 1;
        let mid_mask = (max_qubit_mask - 1) ^ low_mask;
        let high_mask = !(max_qubit_mask - 1);
        // let target_mask1 = 1 << qubit1.index;
        // let target_mask2 = 1 << qubit2.index;
        let target_mask1 = 1usize << qubit2.index;
        let target_mask2 = 1usize << qubit1.index;
        // loop variables
        let dim = 16;
        //let dim = 32;
        println!("min_qubit_mask : {:0>64b}", min_qubit_mask);
        println!("max_qubit_mask : {:0>64b}", max_qubit_mask);
        println!("min_qubit_index: {:0>64b}", min_qubit_index);
        println!("max_qubit_index: {:0>64b}", max_qubit_index);
        println!("low_mask       : {:0>64b}", low_mask);
        println!("mid_mask       : {:0>64b}", mid_mask);
        println!("high_mask      : {:0>64b}", high_mask);
        println!("target_mask1   : {:0>64b}", target_mask1);
        println!("target_mask2   : {:0>64b}", target_mask2);
        for state_index in 0..dim >> qubits_size {
            // create index
            let basis_0 = (state_index & low_mask)
                + ((state_index & mid_mask) << 1)
                + ((state_index & high_mask) << 2);
            println!("state_index        : {:0>4b}", state_index);
            println!(
                "(state_index & low_mask): {:0>4b}",
                (state_index & low_mask)
            );
            println!(
                "((state_index & mid_mask) << 1): {:0>4b}",
                ((state_index & mid_mask) << 1)
            );
            println!(
                "((state_index & high_mask) << 2): {:0>4b}",
                ((state_index & high_mask) << 2)
            );
            // gather index
            let basis_1 = basis_0 + target_mask1;
            let basis_2 = basis_0 + target_mask2;
            let basis_3 = basis_1 + target_mask2;
            println!("{} {} {} {}", basis_0, basis_1, basis_2, basis_3);
        }
    }

    #[test]
    fn test_single_gate2() {
        let dim = 16;
        let qubit1 = &Qubit { index: 1 };
        let qubits = &[qubit1];
        let masks = mask_vec2(qubits);
        println!("mask:{:04b}", masks[0]);
        println!("mask_low:{:04b}", masks[1]);
        println!("mask_high:{:04b}", masks[2]);
        for state_index in 0..dim >> qubits.len() {
            let indices = indices_vec2(state_index, qubits, &masks);
            println!("i[{}] {:?}", state_index, indices);
        }
    }

    #[test]
    fn test_double_gate2() {
        let dim = 16;
        let qubit1 = &Qubit { index: 3 };
        let qubit2 = &Qubit { index: 2 };
        let qubits = &[qubit1, qubit2];
        let masks = mask_vec2(qubits);
        println!("mask:{:04b}", masks[0]);
        println!("mask_low:{:04b}", masks[1]);
        println!("mask_high:{:04b}", masks[2]);
        for state_index in 0..dim >> qubits.len() {
            let indices = indices_vec2(state_index, qubits, &masks);
            println!("i[{}] {:?}", state_index, indices);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::simulator::simulator::{index_pair, indices_vec, mask_pair, mask_vec};
    use crate::Qubit;
    use num::Complex;
    use std::cmp;

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
        let masks = mask_vec(qubits);
        println!("mask:{:04b}", masks[0]);
        println!("mask_low:{:04b}", masks[1]);
        println!("mask_high:{:04b}", masks[2]);
        for state_index in 0..dim >> qubits.len() {
            let indices = indices_vec(state_index, qubits, &masks);
            println!("i[{}] {:?}", state_index, indices);
        }
    }

    #[test]
    fn test_double_gate2() {
        let dim = 16;
        let qubit1 = &Qubit { index: 3 };
        let qubit2 = &Qubit { index: 2 };
        let qubits = &[qubit1, qubit2];
        let masks = mask_vec(qubits);
        println!("mask:{:04b}", masks[0]);
        println!("mask_low:{:04b}", masks[1]);
        println!("mask_high:{:04b}", masks[2]);
        for state_index in 0..dim >> qubits.len() {
            let indices = indices_vec(state_index, qubits, &masks);
            println!("i[{}] {:?}", state_index, indices);
        }
    }

    #[test]
    fn test_mask_pair() {
        let qubit = Qubit { index: 12 };
        let (upper_mask, lower_mask) = mask_pair(&qubit);
        // upper_maskは、index+1が0になる
        assert_eq!(
            upper_mask,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11100000_00000000usize
        );
        // lower_maskは反転
        assert_eq!(
            lower_mask,
            0b00000000_00000000_00000000_00000000_00000000_00000000_00001111_11111111usize
        );

        let qubit = Qubit { index: 0 };
        let (upper_mask, lower_mask) = mask_pair(&qubit);
        println!("qubit index: {}", qubit.index);
        println!("upper_mask: {:0>64b}", upper_mask);
        println!("lower_mask: {:0>64b}", lower_mask);

        let qubit = Qubit { index: 1 };
        let (upper_mask, lower_mask) = mask_pair(&qubit);
        println!("qubit index: {}", qubit.index);
        println!("upper_mask: {:0>64b}", upper_mask);
        println!("lower_mask: {:0>64b}", lower_mask);

        let qubit = Qubit { index: 2 };
        let (upper_mask, lower_mask) = mask_pair(&qubit);
        println!("qubit index: {}", qubit.index);
        println!("upper_mask: {:0>64b}", upper_mask);
        println!("lower_mask: {:0>64b}", lower_mask);
    }

    #[test]
    fn test_index_pair() {
        let qubit = Qubit { index: 13 };
        let (upper_mask, lower_mask) = mask_pair(&qubit);
        let (iz, io) = index_pair(
            0b01011101_11111011_11011111usize,
            &qubit,
            upper_mask,
            lower_mask,
        );
        assert_eq!(iz, 0b10111011_11011011_11011111usize);
        // println!(
        //     "0b10111011_11011011_11011111usize: {}",
        //     0b10111011_11011011_11011111usize
        // );

        assert_eq!(io, 0b10111011_11111011_11011111usize);
        // println!(
        //     "0b10111011_11111011_11011111usize: {}",
        //     0b10111011_11111011_11011111usize
        // );

        // let qubit = Qubit { index: 3 };
        // let (upper_mask, lower_mask) = mask_pair(&qubit);
        // let (iz, io) = index_pair(2, &qubit, upper_mask, lower_mask);
        // println!("iz: {}", iz);
        // println!("io: {}", io);
    }

    #[test]
    fn test_zero_norm_sqr() {
        let states_count = 4;
        let mut states = vec![Complex::new(0., 0.); 1 << states_count];
        states[0] = Complex::new(1., 0.);

        let qubit = Qubit { index: 0 };
        let (upper_mask, lower_mask) = mask_pair(&qubit);
        let zero_norm_sqr: f64 = (0..states.len() >> 1)
            .map(|i| states[index_pair(i, &qubit, upper_mask, lower_mask).0].norm_sqr())
            .sum();
        println!("zero_norm_sqr: {}", zero_norm_sqr);

        for i in 0..states.len() >> 1 {
            println!("i:{} {:0>8b}", i, i);
            let (iz, io) = index_pair(i, &qubit, upper_mask, lower_mask);
            // 2進数 8桁表示
            println!("iz: {} {:0>8b}", iz, iz);
            println!("io: {} {:0>8b}", io, io);
            let state = states[iz];
            println!("state: {}", state);
            let norm_sqr = state.norm_sqr();
            println!("norm_sqr: {}", norm_sqr);
        }
    }
}

// fn apply(&mut self, qubits: &[&Qubit], matrix: &Array2<Complex<f64>>) {
//     let qubits_size = qubits.len();
//     let masks = mask_vec(qubits);
//     for i in 0..self.dim >> qubits_size {
//         let indices = indices_vec(i, qubits, &masks, qubits_size);
//         let values = indices.iter().map(|&i| self.states[i]).collect::<Vec<_>>();
//         let new_values = matrix.dot(&arr1(&values));
//         for (&i, nv) in indices.iter().zip(new_values.to_vec()) {
//             self.states[i] = nv;
//         }
//     }
// }

// fn mask_vec(qubits: &[&Qubit]) -> Vec<usize> {
//     let mut qubits = qubits.to_owned();
//     qubits.sort_by(|a, b| a.index.cmp(&b.index));
//     let mut res = vec![0; qubits.len() + 1];
//     res[0] = 0xFFFF_FFFF_FFFF_FFFFusize << (qubits[qubits.len() - 1].index + 1);
//     for i in 1..qubits.len() {
//         res[i] = (0xFFFF_FFFF_FFFF_FFFFusize << (qubits[qubits.len() - i - 1].index + 1))
//             | (!(0xFFFF_FFFF_FFFF_FFFFusize << (qubits[qubits.len() - i].index)));
//     }
//     res[qubits.len()] = !(0xFFFF_FFFF_FFFF_FFFFusize << qubits[0].index);
//     res
// }

// fn indices_vec(index: usize, qubits: &[&Qubit], mask: &[usize], qubits_size: usize) -> Vec<usize> {
//     let imask = (0..qubits_size + 1)
//         .map(|s| (index << (qubits_size - s)) & mask[s])
//         .fold(0, |acc, m| acc | m);
//     (0..1 << qubits_size)
//         .map(|i| {
//             (0..qubits_size).fold(imask, |acc, j| {
//                 acc | ((i >> (qubits_size - 1 - j) & 0b1) << qubits[j].index)
//             })
//         })
//         .collect()
// }

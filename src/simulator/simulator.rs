use crate::gates::double::DoubleGateApplicator;
use crate::gates::single::SingleGateApplicator;
use crate::gates::triple::TripleGateApplicator;
use crate::{carray, carray_i, MeasuredResult, QuantumMachine, Qubit};
use ndarray::prelude::*;
use num::complex::Complex;
use rand::{self, Rng};
use std::cmp;
pub struct QuantumSimulator {
    dim: usize,
    states: Vec<Complex<f64>>,
}

impl QuantumSimulator {
    pub fn new(n: usize) -> QuantumSimulator {
        let dim = 1 << n;
        let mut states = vec![Complex::new(0., 0.); dim];
        states[0] = Complex::new(1., 0.);

        QuantumSimulator { dim, states }
    }

    fn apply(&mut self, qubits: &[&Qubit], matrix: &Array2<Complex<f64>>) {
        let qubits_size = qubits.len();

        let masks = mask_vec(qubits);

        for i in 0..self.dim >> qubits_size {
            let indices = indices_vec(i, qubits, &masks, qubits_size);
            let values = indices.iter().map(|&i| self.states[i]).collect::<Vec<_>>();
            let new_values = matrix.dot(&arr1(&values));
            for (&i, nv) in indices.iter().zip(new_values.to_vec()) {
                self.states[i] = nv;
            }
        }
    }

    pub fn show(&self) {
        for i in 0..self.dim {
            println!("{:0>4b}> {}", i, self.states[i]);
        }
    }
}

fn mask_vec(qubits: &[&Qubit]) -> Vec<usize> {
    let mut qubits = qubits.to_owned();
    qubits.sort_by(|a, b| a.index.cmp(&b.index));
    let mut res = vec![0; qubits.len() + 1];

    res[0] = 0xFFFF_FFFF_FFFF_FFFFusize << (qubits[qubits.len() - 1].index + 1);

    for i in 1..qubits.len() {
        res[i] = (0xFFFF_FFFF_FFFF_FFFFusize << (qubits[qubits.len() - i - 1].index + 1))
            | (!(0xFFFF_FFFF_FFFF_FFFFusize << (qubits[qubits.len() - i].index)));
    }

    res[qubits.len()] = !(0xFFFF_FFFF_FFFF_FFFFusize << qubits[0].index);

    res
}

fn indices_vec(index: usize, qubits: &[&Qubit], mask: &[usize], qubits_size: usize) -> Vec<usize> {
    let imask = (0..qubits_size + 1)
        .map(|s| (index << (qubits_size - s)) & mask[s])
        .fold(0, |acc, m| acc | m);
    (0..1 << qubits_size)
        .map(|i| {
            (0..qubits_size).fold(imask, |acc, j| {
                acc | ((i >> (qubits_size - 1 - j) & 0b1) << qubits[j].index)
            })
        })
        .collect()
}

pub fn mask_vec2(qubits: &[&Qubit]) -> Vec<usize> {
    let min_qubit_index = qubits.iter().map(|q| q.index).min().unwrap();
    let max_qubit_index = qubits.iter().map(|q| q.index).max().unwrap();
    let min_qubit_mask = 1usize << min_qubit_index;
    let max_qubit_mask = 1usize
        << if qubits.len() > 1 {
            max_qubit_index - 1
        } else {
            max_qubit_index
        };
    let mask_low = min_qubit_mask - 1;
    let mask_high = !(max_qubit_mask - 1);
    let mut res = Vec::with_capacity(3);
    res.push(max_qubit_mask);
    res.push(mask_low);
    res.push(mask_high);
    res
}

pub fn indices_vec2(index: usize, qubits: &[&Qubit], masks: &[usize]) -> Vec<usize> {
    let mut qubits = qubits.to_owned();
    qubits.sort_by(|a, b| a.index.cmp(&b.index));
    let mut res = Vec::with_capacity(qubits.len());
    let mask = masks[0];
    let mask_low = masks[1];
    let mask_high = masks[2];
    let basis_0 = (index & mask_low) + ((index & mask_high) << qubits.len());
    res.push(basis_0);
    // for i in 1..qubits.len() << 1 {
    //     let basis = basis_0 + mask;
    //     res.push(basis);
    // }
    if qubits.len() == 1 {
        let basis_1 = basis_0 + mask;
        res.push(basis_1);
    } else if qubits.len() == 2 {
        let target_mask1 = 1usize << qubits[1].index;
        let target_mask2 = 1usize << qubits[0].index;
        let basis_1 = basis_0 + target_mask1;
        let basis_2 = basis_0 + target_mask2;
        let basis_3 = basis_1 + target_mask2;
        res.push(basis_1);
        res.push(basis_2);
        res.push(basis_3);
    } else {
        // TODO
        unimplemented!();
    }

    res
}

impl QuantumMachine for QuantumSimulator {
    fn measure(&mut self, qubit: &Qubit) -> MeasuredResult {
        let (upper_mask, lower_mask) = mask_pair(qubit);
        // 状態ベクトルを右シフトして量子数にする
        // TODO: 計算結果をキャッシュ（状態ベクトル数が変わらない限り変わらない）
        // norm_sqr: L2
        // sqr(x)が自乗: sqrt(x)が平方根
        // pub fn norm_sqr(&self) -> T {
        //     self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
        // }
        // 0が測定される確率（各量子ビット毎の0が測定される確率の合計）
        let zero_norm_sqr: f64 = (0..self.dim >> 1)
            .map(|i| self.states[index_pair(i, qubit, upper_mask, lower_mask).0].norm_sqr())
            .sum();

        // let seed: [u8; 32] = [0; 32];
        // let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed(seed);
        //if zero_norm_sqr > rng.gen::<f64>() {
        if zero_norm_sqr > rand::random::<f64>() {
            let norm = zero_norm_sqr.sqrt();
            for i in 0..self.dim >> 1 {
                let (iz, io) = index_pair(i, qubit, upper_mask, lower_mask);
                self.states[iz] /= norm;
                self.states[io] = Complex::new(0., 0.);
            }
            MeasuredResult::Zero
        } else {
            let norm = (1. - zero_norm_sqr).sqrt();
            for i in 0..self.dim >> 1 {
                let (iz, io) = index_pair(i, qubit, upper_mask, lower_mask);
                self.states[io] /= norm;
                self.states[iz] = Complex::new(0., 0.);
            }
            MeasuredResult::One
        }
    }

    fn get_qubits(&self) -> Vec<Qubit> {
        (0..self.dim >> 1).map(|x| Qubit { index: x }).collect()
    }
}

fn mask_pair(qubit: &Qubit) -> (usize, usize) {
    let upper_mask = 0xFFFF_FFFF_FFFF_FFFFusize << (qubit.index + 1);
    let lower_mask = !(0xFFFF_FFFF_FFFF_FFFFusize << qubit.index);
    (upper_mask, lower_mask)
}

fn mask_pair2(qubit: &Qubit) -> (usize, usize) {
    let mask = 1 << qubit.index;
    let mask_low = mask - 1;
    let mask_high = !mask_low;
    (mask_high, mask_low)
}

#[inline]
fn index_pair(index: usize, qubit: &Qubit, upper_mask: usize, lower_mask: usize) -> (usize, usize) {
    let index_zero = ((index << 1) & upper_mask) | (index & lower_mask);
    let index_one = index_zero | (1usize << qubit.index);
    (index_zero, index_one)
}

impl SingleGateApplicator for QuantumSimulator {
    // fn apply_single(&mut self, matrix: &Array2<Complex<f64>>, qubit: &Qubit) {
    //     self.apply(&[qubit], matrix);
    // }
    fn apply_single(&mut self, matrix: &Array2<Complex<f64>>, qubit: &Qubit) {
        let mask = 1usize << qubit.index;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        for state_index in 0..self.dim >> 1 {
            let basis_0 = (state_index & mask_low) + ((state_index & mask_high) << 1);
            let basis_1 = basis_0 + mask;
            println!("{} {}", basis_0, basis_1);
            let cval_0 = self.states[basis_0];
            let cval_1 = self.states[basis_1];
            println!("{} {}", cval_0, cval_1);
            // self.states[basis_0] =
            //     matrix.get((0, 0)).unwrap() * cval_0 + matrix.get((0, 1)).unwrap() * cval_1;
            // self.states[basis_1] =
            //     matrix.get((1, 0)).unwrap() * cval_0 + matrix.get((1, 1)).unwrap() * cval_1;
            let new_values = matrix.dot(&array![cval_0, cval_1]);
            self.states[basis_0] = new_values[0];
            self.states[basis_1] = new_values[1];
        }
    }
}

impl DoubleGateApplicator for QuantumSimulator {
    // fn apply_double(&mut self, matrix: &Array2<Complex<f64>>, qubit1: &Qubit, qubit2: &Qubit) {
    //     self.apply(&[qubit1, qubit2], matrix);
    // }
    fn apply_double(&mut self, matrix: &Array2<Complex<f64>>, qubit1: &Qubit, qubit2: &Qubit) {
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
        for state_index in 0..self.dim >> qubits_size {
            // create index
            let basis_0 = (state_index & low_mask)
                + ((state_index & mid_mask) << 1)
                + ((state_index & high_mask) << 2);
            // gather index
            let basis_1 = basis_0 + target_mask1;
            let basis_2 = basis_0 + target_mask2;
            let basis_3 = basis_1 + target_mask2;

            println!("{} {} {} {}", basis_0, basis_1, basis_2, basis_3);

            // fetch values
            let cval_0 = self.states[basis_0];
            let cval_1 = self.states[basis_1];
            let cval_2 = self.states[basis_2];
            let cval_3 = self.states[basis_3];

            // set values
            // println!("{} matrix: {}", state_index, matrix);
            // println!("{} cval: {}", state_index, cval);
            let new_values = matrix.dot(&array![cval_0, cval_1, cval_2, cval_3]);
            // println!("{} new_values: {}", state_index, new_values);
            self.states[basis_0] = new_values[0];
            self.states[basis_1] = new_values[1];
            self.states[basis_2] = new_values[2];
            self.states[basis_3] = new_values[3];
        }
    }
}

impl TripleGateApplicator for QuantumSimulator {
    fn apply_triple(
        &mut self,
        matrix: &Array2<Complex<f64>>,
        qubit1: &Qubit,
        qubit2: &Qubit,
        qubit3: &Qubit,
    ) {
        self.apply(&[qubit1, qubit2, qubit3], matrix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Qubit;

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
    fn test_mask_pair2() {
        let qubit = Qubit { index: 12 };
        let (upper_mask, lower_mask) = mask_pair2(&qubit);
        println!("upper_mask: {:0>64b}", upper_mask);
        println!("lower_mask: {:0>64b}", lower_mask);
        // upper_maskは、index+1が0になる
        assert_eq!(
            upper_mask,
            //0b11111111_11111111_11111111_11111111_11111111_11111111_11100000_00000000usize
            0b11111111_11111111_11111111_11111111_11111111_11111111_11110000_00000000usize
        );
        // lower_maskは反転
        assert_eq!(
            lower_mask,
            0b00000000_00000000_00000000_00000000_00000000_00000000_00001111_11111111usize
        );
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

    #[test]
    fn test_mask_vec() {
        let qubit = Qubit { index: 0 };
        let masks = mask_vec(&[&qubit]);
        println!("masks  : {:?}", masks);
        for mask in &masks {
            //println!("mask: {:02X}", mask);
            println!("mask   : {:04b}", mask);
        }
        let qubit1 = Qubit { index: 0 };
        let qubit2 = Qubit { index: 1 };
        let masks = mask_vec(&[&qubit1, &qubit2]);
        println!("{:?}", masks);
        for i in 0..masks.len() {
            let mask = masks[i];
            //println!("mask[{}]: {:02X}", i, mask);
            println!("mask[{}]: {:04b}", i, mask);
        }
        let qubit1 = Qubit { index: 0 };
        let qubit2 = Qubit { index: 1 };
        let qubit3 = Qubit { index: 2 };
        let masks = mask_vec(&[&qubit1, &qubit2, &qubit3]);
        println!("{:?}", masks);
        for i in 0..masks.len() {
            let mask = masks[i];
            //println!("mask[{}]: {:02X}", i, mask);
            println!("mask[{}]: {:04b}", i, mask);
        }
        let qubit1 = Qubit { index: 1 };
        let qubit2 = Qubit { index: 2 };
        let masks = mask_vec(&[&qubit1, &qubit2]);
        println!("{:?}", masks);
        for i in 0..masks.len() {
            let mask = masks[i];
            //println!("mask[{}]: {:02X}", i, mask);
            println!("mask[{}]: {:04b}", i, mask);
        }
    }

    #[test]
    fn test_mask_vec2() {
        let qubit1 = Qubit { index: 1 };
        let qubit2 = Qubit { index: 3 };
        let qubit3 = Qubit { index: 5 };
        let masks = mask_vec(&[&qubit1, &qubit2, &qubit3]);
        println!("{:?}", masks);
        for i in 0..masks.len() {
            println!("mask[{}]: {:04b}", i, masks[i]);
        }
    }

    #[test]
    fn test_indices_vec() {
        let qubit1 = Qubit { index: 0 };
        let qubit2 = Qubit { index: 1 };
        let qubits = &[&qubit1, &qubit2];

        let dim = qubits.len();
        let mask = mask_vec(qubits);

        // let mut qubits = qubits.to_owned();
        // qubits.sort_by(|a, b| b.index.cmp(&a.index));

        let index = 1;

        let imask = (0..dim + 1)
            .map(|s| (index << (dim - s)) & mask[s])
            .fold(0, |acc, m| acc | m);
        println!("imask: {:?}", imask);
        let result: Vec<usize> = (0..1 << dim)
            .map(|i| {
                (0..dim).fold(imask, |acc, j| {
                    acc | ((i >> (dim - 1 - j) & 0b1) << qubits[j].index)
                    //0 | ((i >> (dim - 1 - j) & 0b1) << qubits[j].index)
                })
            })
            .collect();
        println!("result: {:?}", result);

        for s in 0..dim + 1 {
            let v = (index << (dim - s)) & mask[s];
            println!("(index << (dim - s)): {:0>8b}", (index << (dim - s)));
            println!("mask[{}]: {:0>8b}", s, mask[s]);
            println!("v: {}", v);
        }
        for i in 0..(1 << dim) {
            for j in 0..dim {
                println!(
                    "(i >> (dim - 1 - j) & 0b1): {:0>8b}",
                    (i >> (dim - 1 - j) & 0b1)
                );
                println!("qubits[j].index: {}", qubits[j].index);
                println!(
                    "(i >> (dim - 1 - j) & 0b1) << qubits[{}].index: {}",
                    j,
                    (i >> (dim - 1 - j) & 0b1) << qubits[j].index
                );
            }
        }
    }

    #[test]
    fn test_apply_single() {
        let matrix = carray![[0., 1.], [1., 0.]];
        let qubit = Qubit { index: 3 };
        let states_count = 4;
        let mut states = vec![Complex::new(0., 0.); 1 << states_count];
        states[0] = Complex::new(1., 0.);
        let dim = states.len();
        let loop_dim = dim >> 1;
        let mask = 1 << qubit.index;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        for state_index in 0..loop_dim {
            let basis_0 = (state_index & mask_low) + ((state_index & mask_high) << 1);
            let basis_1 = basis_0 + mask;
            // println!("i:{} iz:{}", state_index, basis_0);
            // println!("i:{} io:{}", state_index, basis_1);
            println!("{} {}", basis_0, basis_1);
            let cval_0 = states[basis_0];
            let cval_1 = states[basis_1];
            //states[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1;
            //states[basis_1] = matrix[2] * cval_0 + matrix[3] * cval_1;
            let new_values = matrix.dot(&array![cval_0, cval_1]);
            //println!("new_values: {}", new_values);
            states[basis_0] = new_values[0];
            states[basis_1] = new_values[1];
        }
    }

    #[test]
    fn test_apply_double() {
        let states_count = 4;
        let mut states = vec![Complex::new(0., 0.); 1 << states_count];
        states[0] = Complex::new(1., 0.);
        //println!("{:?}", states);

        let qubit1 = Qubit { index: 0 };
        let qubit2 = Qubit { index: 1 };
        let qubits = [&qubit1, &qubit2];
        //let qubits = [&qubit1];
        let masks = mask_vec(&qubits);
        // for i in 0..masks.len() {
        //     let mask = masks[i];
        //     println!("mask[{}]: {:02X}", i, mask);
        //     println!("mask[{}]: {:0>64b}", i, mask);
        // }

        // // Hゲート
        // let matrix = carray![[1., 1.], [1., -1.]] / (2f64).sqrt();
        // println!("matrix: {}", matrix);

        // CNOTゲート
        let matrix = carray![
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.]
        ];
        //println!("matrix: {}", matrix);

        let dim = qubits.len();
        for i in 0..(states.len() >> dim) {
            let indices = indices_vec(i, &qubits, &masks, dim);
            println!("indices[{}]: {:?}", i, indices);
            let values = indices.iter().map(|&i| states[i]).collect::<Vec<_>>();
            //println!("values[{}]: {:?}", i, values);
            let new_values = matrix.dot(&arr1(&values));
            //println!("new_values: {}", new_values);
            for (&i, nv) in indices.iter().zip(new_values.to_vec()) {
                println!("nv[{}]: {}", i, nv);
                states[i] = nv;
            }
        }
        println!("{:?}", states);
    }

    #[test]
    fn test_apply_double2() {
        // CNOTゲート
        let matrix = carray![
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.]
        ];
        let states_count = 2;
        let mut states = vec![Complex::new(0., 0.); 1 << states_count];
        states[0] = Complex::new(1., 0.);
        let dim = states.len();
        let qubit1 = Qubit { index: 0 };
        let qubit2 = Qubit { index: 1 };
        let qubits = [&qubit1, &qubit2];
        let min_qubit_index = cmp::min(qubit1.index, qubit2.index);
        let max_qubit_index = cmp::max(qubit1.index, qubit2.index);
        let min_qubit_mask = 1 << min_qubit_index;
        let max_qubit_mask = 1 << (max_qubit_index - 1);
        let low_mask = min_qubit_mask - 1;
        let mid_mask = (max_qubit_mask - 1) ^ low_mask;
        let high_mask = !(max_qubit_mask - 1);
        // let target_mask1 = 1 << qubit1.index;
        // let target_mask2 = 1 << qubit2.index;
        let target_mask1 = 1 << qubit2.index;
        let target_mask2 = 1 << qubit1.index;
        // loop variables
        let loop_dim = dim >> qubits.len();
        for state_index in 0..loop_dim {
            // create index
            let basis_0 = (state_index & low_mask)
                + ((state_index & mid_mask) << 1)
                + ((state_index & high_mask) << 2);
            // gather index
            let basis_1 = basis_0 + target_mask1; // target_mask1 = 2
            let basis_2 = basis_0 + target_mask2; // target_mask2 = 1
            let basis_3 = basis_1 + target_mask2; // target_mask2 = 1

            // println!("i:{} bi0:{}", state_index, basis_0);
            // println!("i:{} bi1:{}", state_index, basis_1);
            // println!("i:{} bi2:{}", state_index, basis_2);
            // println!("i:{} bi3:{}", state_index, basis_3);

            println!("{} {} {} {}", basis_0, basis_1, basis_2, basis_3);

            // fetch values
            let cval_0 = states[basis_0];
            let cval_1 = states[basis_1];
            let cval_2 = states[basis_2];
            let cval_3 = states[basis_3];

            // set values
            // states[basis_0] = matrix.get((0, 0)).unwrap() * cval_0
            //     + matrix.get((0, 1)).unwrap() * cval_1
            //     + matrix.get((0, 2)).unwrap() * cval_2
            //     + matrix.get((0, 3)).unwrap() * cval_3;
            // states[basis_1] = matrix.get((1, 0)).unwrap() * cval_0
            //     + matrix.get((1, 1)).unwrap() * cval_1
            //     + matrix.get((1, 2)).unwrap() * cval_2
            //     + matrix.get((1, 3)).unwrap() * cval_3;
            // states[basis_2] = matrix.get((2, 0)).unwrap() * cval_0
            //     + matrix.get((2, 1)).unwrap() * cval_1
            //     + matrix.get((2, 2)).unwrap() * cval_2
            //     + matrix.get((2, 3)).unwrap() * cval_3;
            // states[basis_3] = matrix.get((3, 0)).unwrap() * cval_0
            //     + matrix.get((3, 1)).unwrap() * cval_1
            //     + matrix.get((3, 2)).unwrap() * cval_2
            //     + matrix.get((3, 3)).unwrap() * cval_3;
            //let cval = &array![cval_0, cval_1, cval_2, cval_3];
            // println!("{} matrix: {}", state_index, matrix);
            // println!("{} cval: {}", state_index, cval);
            let new_values = matrix.dot(&array![cval_0, cval_1, cval_2, cval_3]);
            //println!("{} new_values: {}", state_index, new_values);
            states[basis_0] = new_values[0];
            states[basis_1] = new_values[1];
            states[basis_2] = new_values[2];
            states[basis_3] = new_values[3];
        }
    }
}

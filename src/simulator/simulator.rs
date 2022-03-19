use crate::gates::double::DoubleGateApplicator;
use crate::gates::single::SingleGateApplicator;
use crate::gates::triple::TripleGateApplicator;
use crate::{MeasuredResult, QuantumMachine, Qubit};
use ndarray::prelude::*;
use num::complex::Complex;
use rand::{self, Rng};

pub struct QuantumSimulator {
    dimension: usize,
    states: Vec<Complex<f64>>,
}

impl QuantumSimulator {
    pub fn new(n: usize) -> QuantumSimulator {
        let mut states = vec![Complex::new(0., 0.); 1 << n];
        states[0] = Complex::new(1., 0.);

        QuantumSimulator {
            dimension: n,
            states: states,
        }
    }

    fn apply(&mut self, qubits: &[&Qubit], matrix: &Array2<Complex<f64>>) {
        let dim = qubits.len();

        let masks = mask_vec(qubits);

        for i in 0..(self.states.len() >> dim) {
            let indices = indices_vec(i, qubits, &masks, dim);
            let values = indices.iter().map(|&i| self.states[i]).collect::<Vec<_>>();
            let new_values = matrix.dot(&arr1(&values));
            for (&i, nv) in indices.iter().zip(new_values.to_vec()) {
                self.states[i] = nv;
            }
        }
    }

    pub fn show(&self) {
        for i in 0..self.states.len() {
            println!("{}", self.states[i]);
        }
    }
}

fn mask_vec(qubits: &[&Qubit]) -> Vec<usize> {
    let mut qubits = qubits.to_owned();
    qubits.sort_by(|a, b| a.index.cmp(&b.index));
    let mut res = vec![0; qubits.len() + 1];

    // 最後のqubitsのindex+1分を左シフトする(upper_mask的)
    res[0] = 0xFFFF_FFFF_FFFF_FFFFusize << (qubits[qubits.len() - 1].index + 1);

    for i in 1..qubits.len() {
        // 後ろから見ていく
        // bitのorを取る
        res[i] = (0xFFFF_FFFF_FFFF_FFFFusize << (qubits[qubits.len() - i - 1].index + 1))
            | (!(0xFFFF_FFFF_FFFF_FFFFusize << (qubits[qubits.len() - i].index)));
    }

    res[qubits.len()] = !(0xFFFF_FFFF_FFFF_FFFFusize << qubits[0].index);

    res
}

fn indices_vec(index: usize, qubits: &[&Qubit], mask: &[usize], dim: usize) -> Vec<usize> {
    let imask = (0..dim + 1)
        .map(|s| (index << (dim - s)) & mask[s])
        .fold(0, |acc, m| acc | m);
    (0..1 << dim)
        .map(|i| {
            (0..dim).fold(imask, |acc, j| {
                acc | ((i >> (dim - 1 - j) & 0b1) << qubits[j].index)
            })
        })
        .collect()
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
        let zero_norm_sqr: f64 = (0..(self.states.len() >> 1))
            .map(|i| self.states[index_pair(i, qubit, upper_mask, lower_mask).0].norm_sqr())
            .sum();

        // let seed: [u8; 32] = [0; 32];
        // let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed(seed);
        //if zero_norm_sqr > rng.gen::<f64>() {
        if zero_norm_sqr > rand::random::<f64>() {
            let norm = zero_norm_sqr.sqrt();
            for i in 0..(self.states.len() >> 1) {
                let (iz, io) = index_pair(i, qubit, upper_mask, lower_mask);
                self.states[iz] /= norm;
                self.states[io] = Complex::new(0., 0.);
            }
            MeasuredResult::Zero
        } else {
            let norm = (1. - zero_norm_sqr).sqrt();
            for i in 0..(self.states.len() >> 1) {
                let (iz, io) = index_pair(i, qubit, upper_mask, lower_mask);
                self.states[io] /= norm;
                self.states[iz] = Complex::new(0., 0.);
            }
            MeasuredResult::One
        }
    }

    fn get_qubits(&self) -> Vec<Qubit> {
        (0..self.dimension).map(|x| Qubit { index: x }).collect()
    }
}

fn mask_pair(qubit: &Qubit) -> (usize, usize) {
    let upper_mask = 0xFFFF_FFFF_FFFF_FFFFusize << (qubit.index + 1);
    let lower_mask = !(0xFFFF_FFFF_FFFF_FFFFusize << qubit.index);
    (upper_mask, lower_mask)
}

#[inline]
fn index_pair(index: usize, qubit: &Qubit, upper_mask: usize, lower_mask: usize) -> (usize, usize) {
    let index_zero = ((index << 1) & upper_mask) | (index & lower_mask);
    let index_one = index_zero | (1usize << qubit.index);
    (index_zero, index_one)
}

impl SingleGateApplicator for QuantumSimulator {
    fn apply_single(&mut self, matrix: &Array2<Complex<f64>>, qubit: &Qubit) {
        self.apply(&[qubit], matrix);
    }
}

impl DoubleGateApplicator for QuantumSimulator {
    fn apply_double(&mut self, matrix: &Array2<Complex<f64>>, qubit1: &Qubit, qubit2: &Qubit) {
        self.apply(&[qubit1, qubit2], matrix);
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

        assert_eq!(io, 0b10111011_11111011_11011111usize);
    }

    #[test]
    fn test_zero_norm_sqr() {
        let states_count = 4;
        let mut states = vec![Complex::new(0., 0.); 1 << states_count];
        states[0] = Complex::new(1., 0.);

        let qubit = Qubit { index: 0 };
        let (upper_mask, lower_mask) = mask_pair(&qubit);
        let zero_norm_sqr: f64 = (0..(states.len() >> 1))
            .map(|i| states[index_pair(i, &qubit, upper_mask, lower_mask).0].norm_sqr())
            .sum();
        println!("zero_norm_sqr: {}", zero_norm_sqr);

        for i in 0..(states.len() >> 1) {
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
        println!("masks: {:?}", masks);
        for mask in &masks {
            println!("mask: {:02X}", *mask);
        }
        let qubit1 = Qubit { index: 0 };
        let qubit2 = Qubit { index: 1 };
        let masks = mask_vec(&[&qubit1, &qubit2]);
        println!("{:?}", masks);
        for i in 0..masks.len() {
            let mask = masks[i];
            println!("mask[{}]: {:02X}", i, mask);
        }
        let qubit1 = Qubit { index: 0 };
        let qubit2 = Qubit { index: 1 };
        let qubit3 = Qubit { index: 2 };
        let masks = mask_vec(&[&qubit1, &qubit2, &qubit3]);
        println!("{:?}", masks);
        for i in 0..masks.len() {
            let mask = masks[i];
            println!("mask[{}]: {:02X}", i, mask);
        }
        let qubit1 = Qubit { index: 2 };
        let qubit2 = Qubit { index: 1 };
        let masks = mask_vec(&[&qubit1, &qubit2]);
        println!("{:?}", masks);
        for i in 0..masks.len() {
            let mask = masks[i];
            println!("mask[{}]: {:02X}", i, mask);
        }
    }

    #[test]
    fn test_index_vec() {
        // let dim = qubits.len();
        // let masks = mask_vec(qubits);
        // for i in 0..(self.states.len() >> dim) {
        //     let indices = indices_vec(i, qubits, &masks, dim);
        //     let values = indices.iter().map(|&i| self.states[i]).collect::<Vec<_>>();
        //     let new_values = matrix.dot(&arr1(&values));
        //     for (&i, nv) in indices.iter().zip(new_values.to_vec()) {
        //         self.states[i] = nv;
        //     }
        // }

        // fn indices_vec(index: usize, qubits: &[&Qubit], mask: &[usize], dim: usize) -> Vec<usize> {
        //     let imask = (0..dim + 1)
        //         .map(|s| (index << (dim - s)) & mask[s])
        //         .fold(0, |acc, m| acc | m);
        //     (0..1 << dim)
        //         .map(|i| {
        //             (0..dim).fold(imask, |acc, j| {
        //                 acc | ((i >> (dim - 1 - j) & 0b1) << qubits[j].index)
        //             })
        //         })
        //         .collect()
        // }

        let states_count = 2;
        let mut states = vec![Complex::new(0., 0.); 1 << states_count];
        //states[0] = Complex::new(1., 0.);
        states[1] = Complex::new(1., 0.);
        println!("{:?}", states);

        let qubit1 = Qubit { index: 0 };
        let qubit2 = Qubit { index: 1 };
        let qubits = [&qubit1, &qubit2];
        //let qubits = [&qubit1];
        let masks = mask_vec(&qubits);
        for i in 0..masks.len() {
            let mask = masks[i];
            println!("mask[{}]: {:02X}", i, mask);
            println!("mask[{}]: {:0>64b}", i, mask);
        }

        // Hゲート
        // macro_rules! carray {
        //     ( $([$($x: expr),*]),* ) => {{
        //         use num::complex::Complex;
        //         array![
        //             $([$(Complex::new($x, 0.)),*]),*
        //         ]
        //     }};
        // }
        // let matrix = carray![[1., 1.], [1., -1.]] / (2f64).sqrt();
        // println!("matrix: {}", matrix);

        macro_rules! carray {
            ( $([$($x: expr),*]),* ) => {{
                use num::complex::Complex;
                array![
                    $([$(Complex::new($x, 0.)),*]),*
                ]
            }};
        }

        let dim = qubits.len();
        // ゲートの量子数分右シフトする
        for i in 0..(states.len() >> dim) {
            let indices = indices_vec(i, &qubits, &masks, dim);
            println!("indices[{}]: {:?}", i, indices);
            let values = indices.iter().map(|&i| states[i]).collect::<Vec<_>>();
            println!("values[{}]: {:?}", i, values);
            // let new_values = matrix.dot(&arr1(&values));
            // println!("new_values: {}", new_values);
            // for (&i, nv) in indices.iter().zip(new_values.to_vec()) {
            //     println!("nv[{}]: {}", i, nv);
            //     states[i] = nv;
            // }
        }
        //        println!("{:?}", states);
    }
}

use crate::gates::double::DoubleGateApplicator;
use crate::gates::single::SingleGateApplicator;
use crate::gates::triple::TripleGateApplicator;
use crate::{MeasuredResult, QuantumMachine, Qubit};
use ndarray::prelude::*;
use num::complex::Complex;
use rand::{self};
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
        let masks = mask_vec(qubits);
        println!("masks: {:?}", masks);
        for i in 0..self.dim >> qubits.len() {
            let indices = indices_vec(i, qubits, &masks);
            println!("indices_vec: {:?}", indices);
            let values = indices.iter().map(|&i| self.states[i]).collect::<Vec<_>>();
            let new_values = matrix.dot(&arr1(&values));
            for (&i, nv) in indices.iter().zip(new_values.to_vec()) {
                self.states[i] = nv;
            }
        }
    }

    // TODO: applyと統合
    fn apply_with_ctrl(
        &mut self,
        qubits_ctrl: &[&Qubit],
        qubits: &[&Qubit],
        matrix: &Array2<Complex<f64>>,
    ) {
        let masks = mask_vec_with_ctrl(qubits_ctrl, qubits);
        println!("masks: {:?}", masks);
        let qsize = qubits_ctrl.len() + qubits.len();
        for i in 0..self.dim >> qsize {
            let indices = indices_vec_with_ctrl(i, qubits_ctrl, qubits, &masks);
            println!("indices_vec_with_ctrl: {:?}", indices);
            let values = indices.iter().map(|&i| self.states[i]).collect::<Vec<_>>();
            // 制御ビットがついて4x1の行列になるため、回路側の行列を併せる
            // flattenで1次元にしてしまってよい？
            // let fmatrix = Array::from_iter(matrix.iter()); //matrix.flatten();
            // let new_values = fmatrix.dot(&arr1(&values));
            // for (&i, nv) in indices.iter().zip(new_values.to_vec()) {
            //     self.states[i] = nv;
            // }

            // TODO: 取り敢えず1制御ビットと1回路固定で実装する
            self.states[indices[0]] =
                matrix.get((0, 0)).unwrap() * values[0] + matrix.get((0, 1)).unwrap() * values[1];
            self.states[indices[1]] =
                matrix.get((1, 0)).unwrap() * values[2] + matrix.get((1, 1)).unwrap() * values[3];
        }
    }

    pub fn show(&self) {
        for i in 0..self.dim {
            println!("{:0>4b}> {}", i, self.states[i]);
        }
    }
}

pub fn mask_vec(qubits: &[&Qubit]) -> Vec<usize> {
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

pub fn indices_vec(index: usize, qubits: &[&Qubit], masks: &[usize]) -> Vec<usize> {
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

pub fn mask_vec_with_ctrl(qubits_ctl: &[&Qubit], qubits_tgt: &[&Qubit]) -> Vec<usize> {
    let mut qubits = qubits_ctl.to_vec();
    qubits.append(&mut qubits_tgt.to_vec());

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

// TODO: indices_vec2と統合
// TODO: たんなるシングルゲートになっている。。（制御が効いていない。。）
pub fn indices_vec_with_ctrl(
    index: usize,
    qubits_ctl: &[&Qubit],
    qubits_tgt: &[&Qubit],
    masks: &[usize],
) -> Vec<usize> {
    //    let mut qubits = qubits.to_owned();
    let mut qubits = qubits_ctl.to_vec();
    qubits.append(&mut qubits_tgt.to_vec());

    qubits.sort_by(|a, b| a.index.cmp(&b.index));
    let mut res = Vec::with_capacity(qubits.len());
    let mask = masks[0];
    let mask_low = masks[1];
    let mask_high = masks[2];
    let control_mask = 1usize << qubits_ctl[0].index;
    let basis_0 = (index & mask_low) + ((index & mask_high) << qubits.len()) * control_mask;
    res.push(basis_0);
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

pub fn mask_pair(qubit: &Qubit) -> (usize, usize) {
    let upper_mask = 0xFFFF_FFFF_FFFF_FFFFusize << (qubit.index + 1);
    let lower_mask = !(0xFFFF_FFFF_FFFF_FFFFusize << qubit.index);
    (upper_mask, lower_mask)
}

#[inline]
pub fn index_pair(
    index: usize,
    qubit: &Qubit,
    upper_mask: usize,
    lower_mask: usize,
) -> (usize, usize) {
    let index_zero = ((index << 1) & upper_mask) | (index & lower_mask);
    let index_one = index_zero | (1usize << qubit.index);
    (index_zero, index_one)
}

impl SingleGateApplicator for QuantumSimulator {
    fn apply_single(&mut self, matrix: &Array2<Complex<f64>>, qubit: &Qubit) {
        self.apply(&[qubit], matrix);
    }
    fn apply_single_with_ctrl(
        &mut self,
        matrix: &Array2<Complex<f64>>,
        qubit_ctrl: &Qubit,
        qubit: &Qubit,
    ) {
        self.apply_with_ctrl(&[qubit_ctrl], &[qubit], matrix);
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

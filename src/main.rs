extern crate rusqx;

use rusqx::prelude::*;

fn set(sim: &mut QuantumSimulator, qubit: &Qubit, r: MeasuredResult) {
    if sim.measure(qubit) != r {
        sim.X(qubit);
    }
}

fn main() {
    let mut sim = QuantumSimulator::new(2);
    let qubits = sim.get_qubits();
    let measure_count = 10000;

    for _ in 0..measure_count {
        set(&mut sim, &qubits[0], MeasuredResult::Zero);
        set(&mut sim, &qubits[1], MeasuredResult::Zero);

        sim.H(&qubits[0]);
        sim.CNOT(&qubits[0], &qubits[1]);

        assert_eq!(sim.measure(&qubits[0]), sim.measure(&qubits[1]));
    }
}

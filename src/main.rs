extern crate rusqx;

use rusqx::prelude::*;

fn set(sim: &mut QuantumSimulator, qubit: &Qubit, r: MeasuredResult) {
    if sim.measure(qubit) != r {
        sim.X(qubit);
    }
}

fn main() {
    let mut sim = QuantumSimulator::new(4);
    let qubits = sim.get_qubits();
    //let measure_count = 10000;
    let measure_count = 1;

    for _ in 0..measure_count {
        set(&mut sim, &qubits[0], MeasuredResult::Zero);
        set(&mut sim, &qubits[1], MeasuredResult::Zero);
        set(&mut sim, &qubits[2], MeasuredResult::Zero);
        set(&mut sim, &qubits[3], MeasuredResult::Zero);

        // sim.X(&qubits[0]);
        // sim.X(&qubits[1]);
        // sim.X(&qubits[2]);

        // sim.H(&qubits[1]);
        // sim.show();
        // sim.H(&qubits[2]);

        // sim.X(&qubits[0]);
        // sim.CNOT(&qubits[0], &qubits[1]);

        sim.X(&qubits[0]);
        sim.X_C(&qubits[0], &qubits[1]);

        // sim.X(&qubits[1]);
        // sim.CNOT(&qubits[1], &qubits[0]);

        // sim.X(&qubits[2]);
        // sim.CNOT(&qubits[2], &qubits[1]);

        // sim.X(&qubits[2]);
        // sim.CNOT(&qubits[2], &qubits[3]);

        //assert_eq!(sim.measure(&qubits[0]), sim.measure(&qubits[1]));

        println!("--------");
        sim.show();
    }
}

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np

def run_test_case(error_qubit, shots=10000, apply_final_h=True):
    data = QuantumRegister(3, 'data')
    ancilla = QuantumRegister(2, 'ancilla')
    syndrome = ClassicalRegister(2, 'syndrome')
    output = ClassicalRegister(1, 'output')
    qc = QuantumCircuit(data, ancilla, syndrome, output)

    alpha, beta = np.sqrt(0.7), np.sqrt(0.3)
    qc.initialize([alpha, beta], data[0])
    qc.h(data[0])
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    qc.h(data)

    print(f"\nTest Case: {error_qubit}")
    if error_qubit != 'none':
        print(f"Injecting phase-flip error on {error_qubit}")
        if error_qubit == 'data[0]':
            qc.z(data[0])
        elif error_qubit == 'data[1]':
            qc.z(data[1])
        else:
            qc.z(data[2])
    else:
        print("No error injected")

    qc.h(data)
    qc.reset(ancilla)
    qc.cx(data[0], ancilla[0])
    qc.cx(data[1], ancilla[0])
    qc.cx(data[1], ancilla[1])
    qc.cx(data[2], ancilla[1])
    qc.barrier()
    qc.measure(ancilla, syndrome)

    with qc.if_test((syndrome, 0b10)):  # q2 error
        qc.z(data[2])
    with qc.if_test((syndrome, 0b01)):  # q0 error
        qc.z(data[0])
    with qc.if_test((syndrome, 0b11)):
        qc.z(data[1])

    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    if apply_final_h:
        qc.h(data[0])
    qc.measure(data[0], output)

    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Reverse syndrome bits
    reversed_counts = {}
    for key, value in counts.items():
        if ' ' in key:
            output_bit, syndrome_bits = key.split()
            reversed_syndrome = syndrome_bits[::-1]
            new_key = f"{output_bit} {reversed_syndrome}"
        else:
            new_key = key
        reversed_counts[new_key] = value

    output_counts = {'0': 0, '1': 0}
    for key in reversed_counts:
        output_bit = key.split()[0] if ' ' in key else key[-1]
        output_counts[output_bit] += reversed_counts[key]

    print("Syndrome counts:", {k: v for k, v in reversed_counts.items() if len(k) >= 2})
    print("Final output counts:", output_counts)
    return reversed_counts, output_counts

test_cases = ['data[0]', 'data[1]', 'data[2]', 'none']
shots = 10000
apply_final_h = True

for test_case in test_cases:
    counts, output_counts = run_test_case(test_case, shots, apply_final_h)

print("\nRunning no-error case without final H gate to test output mapping:")
run_test_case('none', shots, apply_final_h=False)

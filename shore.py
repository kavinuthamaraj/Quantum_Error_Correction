from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import numpy as np
import random

# User-defined parameters
alpha = np.sqrt(0.2)  # |0> state
beta = np.sqrt(0.8)
bit_flip_prob = 0.5
phase_flip_prob = 0.5

# Check normalization
norm = abs(alpha)**2 + abs(beta)**2
if not np.isclose(norm, 1.0, atol=1e-6):
    raise ValueError(f"State not normalized: |alpha|^2 + |beta|^2 = {norm}, expected 1.0")

# Initialize registers
data = QuantumRegister(9, 'data')
bit_syndrome = QuantumRegister(6, 'bit_syndrome')
phase_syndrome = QuantumRegister(2, 'phase_syndrome')
bit_meas = ClassicalRegister(6, 'bit_meas')
phase_meas = ClassicalRegister(2, 'phase_meas')
output_meas = ClassicalRegister(1, 'output')
qc = QuantumCircuit(data, bit_syndrome, phase_syndrome, bit_meas, phase_meas, output_meas)

# Step 1: Initialize logical qubit with alpha|0> + beta|1>
qc.initialize([alpha, beta], data[0])

# Step 2: Encode the logical qubit into Shor's 9-qubit code
# Phase-flip encoding
qc.cx(data[0], data[3])
qc.cx(data[0], data[6])
qc.h(data[0])
qc.h(data[3])
qc.h(data[6])
# Bit-flip encoding
qc.cx(data[0], data[1])
qc.cx(data[0], data[2])
qc.cx(data[3], data[4])
qc.cx(data[3], data[5])
qc.cx(data[6], data[7])
qc.cx(data[6], data[8])

qc.barrier()

# Step 3: Simulate random errors
#error_qubit = random.randint(0, 8)
error_qubit = 8
error_info = f"Errors introduced on data[{error_qubit}]: "
bit_flip = random.random() < bit_flip_prob
phase_flip = random.random() < phase_flip_prob
if bit_flip:
    qc.x(data[error_qubit])
    error_info += "Bit-flip (X) "
if phase_flip:
    qc.z(data[error_qubit])
    error_info += "Phase-flip (Z) "
if not (bit_flip or phase_flip):
    error_info += "None"

# Extra- One more error on Y
print(error_info)

qc.barrier()

# Step 4: Syndrome measurement for bit-flip errors
# Reset syndrome qubits

#qc.reset(bit_syndrome)

# Block 1: data[0,1,2]
qc.cx(data[0], bit_syndrome[0])
qc.cx(data[2], bit_syndrome[0])
qc.cx(data[1], bit_syndrome[1])
qc.cx(data[2], bit_syndrome[1])
# Block 2: data[3,4,5]
qc.cx(data[3], bit_syndrome[2])
qc.cx(data[5], bit_syndrome[2])
qc.cx(data[4], bit_syndrome[3])
qc.cx(data[5], bit_syndrome[3])

# Block 3: data[6,7,8]
qc.cx(data[6], bit_syndrome[4])
qc.cx(data[8], bit_syndrome[4])
qc.cx(data[7], bit_syndrome[5])
qc.cx(data[8], bit_syndrome[5])

# Measure bit-flip syndromes
qc.measure(bit_syndrome[0], bit_meas[0])
qc.measure(bit_syndrome[1], bit_meas[1])
qc.measure(bit_syndrome[2], bit_meas[2])
qc.measure(bit_syndrome[3], bit_meas[3])
qc.measure(bit_syndrome[4], bit_meas[4])
qc.measure(bit_syndrome[5], bit_meas[5])


# Correct bit-flip errors
# Block 1: 10 -> flip data[0], 01 -> flip data[2], 11 -> flip data[1]
with qc.if_test((bit_meas, 0b10)) as else_:
    qc.x(data[0])
with else_:
    with qc.if_test((bit_meas, 0b01)):
        qc.x(data[1])
    with qc.if_test((bit_meas, 0b11)):
        qc.x(data[2])
# Block 2: 10 -> flip data[3], 01 -> flip data[5], 11 -> flip data[4]
with qc.if_test((bit_meas, 0b10)) as else_:
    qc.x(data[3])
with else_:
    with qc.if_test((bit_meas, 0b01)):
        qc.x(data[4])
    with qc.if_test((bit_meas, 0b11)):
        qc.x(data[5])

# Block 2: 10 -> flip data[3], 01 -> flip data[5], 11 -> flip data[4]
with qc.if_test((bit_meas, 0b10)) as else_:
    qc.x(data[6])
with else_:
    with qc.if_test((bit_meas, 0b01)):
        qc.x(data[7])
    with qc.if_test((bit_meas, 0b11)):
        qc.x(data[8])


qc.barrier()

# Step 5: Syndrome measurement for phase-flip errors
# Reset syndrome qubits
#qc.reset(phase_syndrome)

qc.h(data[0])
qc.h(data[1])
qc.h(data[2])
qc.h(data[3])
qc.h(data[4])
qc.h(data[5])
qc.h(data[6])
qc.h(data[7])
qc.h(data[8])

qc.cx(data[0], phase_syndrome[0])
qc.cx(data[1], phase_syndrome[0])
qc.cx(data[2], phase_syndrome[0])
qc.cx(data[3], phase_syndrome[0])
qc.cx(data[4], phase_syndrome[0])
qc.cx(data[5], phase_syndrome[0])

qc.cx(data[3], phase_syndrome[1])
qc.cx(data[4], phase_syndrome[1])
qc.cx(data[5], phase_syndrome[1])
qc.cx(data[6], phase_syndrome[1])
qc.cx(data[7], phase_syndrome[1])
qc.cx(data[8], phase_syndrome[1])

qc.cx(phase_syndrome[0],phase_syndrome[1])
# Transform back

qc.h(data[0])
qc.h(data[1])
qc.h(data[2])
qc.h(data[3])
qc.h(data[4])
qc.h(data[5])
qc.h(data[6])
qc.h(data[7])
qc.h(data[8])

# Measure phase differences between blocks
qc.measure(phase_syndrome[0], phase_meas[0])
qc.measure(phase_syndrome[1], phase_meas[1])


'''
# Correct phase-flip errors
# 10 -> flip data[0], 01 -> flip data[6], 11 -> flip data[3]
with qc.if_test((phase_meas, 0b10)) as else_:
    qc.z(data[0])
with else_:
    with qc.if_test((phase_meas, 0b01)):
        qc.z(data[6])
    with qc.if_test((phase_meas, 0b11)):
        qc.z(data[3])
'''

# Step 5: Syndrome measurement for phase-flip errors
qc.h(data[0])
qc.h(data[1])
qc.h(data[2])
qc.h(data[3])
qc.h(data[4])
qc.h(data[5])
qc.h(data[6])
qc.h(data[7])
qc.h(data[8])

qc.cx(data[0], phase_syndrome[0])
qc.cx(data[1], phase_syndrome[0])
qc.cx(data[2], phase_syndrome[0])
qc.cx(data[3], phase_syndrome[0])
qc.cx(data[4], phase_syndrome[0])
qc.cx(data[5], phase_syndrome[0])

qc.cx(data[3], phase_syndrome[1])
qc.cx(data[4], phase_syndrome[1])
qc.cx(data[5], phase_syndrome[1])
qc.cx(data[6], phase_syndrome[1])
qc.cx(data[7], phase_syndrome[1])
qc.cx(data[8], phase_syndrome[1])

qc.h(data[0])
qc.h(data[1])
qc.h(data[2])
qc.h(data[3])
qc.h(data[4])
qc.h(data[5])
qc.h(data[6])
qc.h(data[7])
qc.h(data[8])

# Measure phase differences between blocks
qc.measure(phase_syndrome[0], phase_meas[0])
qc.measure(phase_syndrome[1], phase_meas[1])

# Correct phase-flip errors - APPLY TO ALL QUBITS IN THE AFFECTED BLOCK
with qc.if_test((phase_meas, 0b10)):  # First block has phase flip
    qc.z(data[0])
    qc.z(data[1])
    qc.z(data[2])
with qc.if_test((phase_meas, 0b01)):  # Third block has phase flip
    qc.z(data[6])
    qc.z(data[7])
    qc.z(data[8])
with qc.if_test((phase_meas, 0b11)):  # Second block has phase flip
    qc.z(data[3])
    qc.z(data[4])
    qc.z(data[5])


qc.barrier()

# Step 6: Decode the logical qubit
# Reverse bit-flip encoding
qc.cx(data[0], data[2])
qc.cx(data[0], data[1])
qc.cx(data[3], data[5])
qc.cx(data[3], data[4])
qc.cx(data[6], data[8])
qc.cx(data[6], data[7])
# Reverse phase-flip encoding
qc.h(data[6])
qc.h(data[3])
qc.h(data[0])
qc.cx(data[0], data[6])
qc.cx(data[0], data[3])



# Measure the logical qubit
qc.measure(data[0], output_meas[0])


# Execute the circuit
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
result = simulator.run(compiled_circuit, shots=1024).result()
counts = result.get_counts()

# Count 0s and 1s for the logical qubit
zero_count = sum(v for k, v in counts.items() if k.split()[0] == '0')
one_count = sum(v for k, v in counts.items() if k.split()[0] == '1')




# Print results
print("Measurement outcomes:", counts)
print(f"Number of 0s: {zero_count}")
print(f"Number of 1s: {one_count}")

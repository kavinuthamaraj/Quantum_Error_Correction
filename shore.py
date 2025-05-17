from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np
import random

print('\nShor Code with Manual Error Testing')
print('-----------------------------------')

# Define custom alpha and beta
alpha = np.sqrt(0.9)  # |0> amplitude
beta = np.sqrt(0.1)   # |1> amplitude
print(f'Initial state: alpha = {alpha:.3f}, beta = {beta:.3f}')
print(f'Expected probabilities - |0>: {abs(alpha)**2:.3f}, |1>: {abs(beta)**2:.3f}')

# Error probabilities for random errors
p_bit = 0.01   # Probability of bit flip per qubit
p_phase = 0.01 # Probability of phase flip per qubit

# Simple circuit with randomized errors (without correction)
qr = QuantumRegister(1, 'q')
cr = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qr, cr)

# Initialize custom state
theta = 2 * np.arccos(alpha)
circuit.u(theta, 0, 0, 0)

# Apply randomized errors
bit_flip = random.random() < p_bit
phase_flip = random.random() < p_phase
if bit_flip:
    circuit.x(0)
if phase_flip:
    circuit.z(0)
print("\nUncorrected circuit errors:")
print(f"Bit flip on q[0]: {bit_flip}")
print(f"Phase flip on q[0]: {phase_flip}")

circuit.barrier()
circuit.measure(0, 0)

# Run on simulator
simulator = AerSimulator()
compiled_circuit = transpile(circuit, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts()

print("\nUncorrected randomized errors")
print("-----------------------------")
print(f"Counts: {counts}")

# Shor code circuit
qr = QuantumRegister(9, 'q')
cr = ClassicalRegister(1, 'c')
shor_circuit = QuantumCircuit(qr, cr)

# Initialize custom state
shor_circuit.u(theta, 0, 0, 0)

# Encoding
shor_circuit.cx(0, 3)
shor_circuit.cx(0, 6)
shor_circuit.h([0, 3, 6])
shor_circuit.cx(0, 1)
shor_circuit.cx(3, 4)
shor_circuit.cx(6, 7)
shor_circuit.cx(0, 2)
shor_circuit.cx(3, 5)
shor_circuit.cx(6, 8)
shor_circuit.barrier()

# Apply errors (manual or random)
print("\nShor code errors:")
# Manual error insertion: Modify this section for test cases
# Example: shor_circuit.x(1) for bit flip on q[1]
# Example: shor_circuit.z(3) for phase flip on q[3]
# For random errors (default):
bit_flips = [random.random() < p_bit for _ in range(9)]
phase_flips = [random.random() < p_phase for _ in range(9)]
error_count = 0
for i in range(9):
    if bit_flips[i]:
        shor_circuit.x(i)
        print(f"Bit flip on q[{i}]")
        error_count += 1
    if phase_flips[i]:
        shor_circuit.z(i)
        print(f"Phase flip on q[{i}]")
        error_count += 1
if error_count == 0:
    print("No errors applied")

shor_circuit.barrier()

# Error correction
shor_circuit.cx(0, 1)
shor_circuit.cx(3, 4)
shor_circuit.cx(6, 7)
shor_circuit.cx(0, 2)
shor_circuit.cx(3, 5)
shor_circuit.cx(6, 8)
shor_circuit.ccx(1, 2, 0)
shor_circuit.ccx(4, 5, 3)
shor_circuit.ccx(8, 7, 6)
shor_circuit.h([0, 3, 6])
shor_circuit.cx(0, 3)
shor_circuit.cx(0, 6)
shor_circuit.ccx(6, 3, 0)
shor_circuit.barrier()

# Measurement
shor_circuit.measure(0, 0)

# Run on simulator
compiled_shor = transpile(shor_circuit, simulator)
job_shor = simulator.run(compiled_shor, shots=1000)
result_shor = job_shor.result()
counts_shor = result_shor.get_counts()

# Verify success
expected_0 = 1000 * abs(alpha)**2  # ~700
expected_1 = 1000 * abs(beta)**2   # ~300
count_0 = counts_shor.get('0', 0)
count_1 = counts_shor.get('1', 0)
tolerance = 100
success = (abs(count_0 - expected_0) < tolerance and abs(count_1 - expected_1) < tolerance)

print("\nShor code with errors")
print("---------------------")
print(f"Counts: {counts_shor}")
print(f"Expected if corrected: ~{int(expected_0)} |0>, ~{int(expected_1)} |1>")
print(f"Total errors applied: {error_count}")
print(f"Success (within ±10%): {'Yes' if success else 'No'}")

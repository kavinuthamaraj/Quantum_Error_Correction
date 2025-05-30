from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
import random

#Step 1 - Intialising Registers
data = QuantumRegister(3,'data')
ancilla = QuantumRegister(2,'ancilla')
syndrome = ClassicalRegister(2,'syndrome')
output = ClassicalRegister(1,'Output')
qc = QuantumCircuit(data,ancilla,syndrome,output)

#Step 2 - Encoding
alpha, beta = np.sqrt(0.7), np.sqrt(0.3)
qc.initialize([alpha,beta],data[0])
qc.cx(data[0],data[1])
qc.cx(data[0],data[2])



#Step 3 - Error inducing
error_qubit = random.randint(0,3)
if error_qubit < 3:
    qc.x(data[error_qubit])
    print(f"Error injected in qubit data[{error_qubit}]")

else:
    print("No error injected")


#Step 4 - Reciever - Detecting and C
qc.cx(data[0],ancilla[0])
qc.cx(data[2],ancilla[0])
qc.cx(data[1], ancilla[1])
qc.cx(data[2], ancilla[1])
qc.barrier()
qc.measure(ancilla[0], syndrome[0]) # syndrome[0] = q0 XOR q2 (LSB) 
qc.measure(ancilla[1], syndrome[1]) # syndrome[1] = q1 XOR q2 (MSB)

#Step 5 Error Correction
with qc.if_test((syndrome, 0b01)):  # '01' → Error on data[0]
    qc.x(data[0])
with qc.if_test((syndrome, 0b10)):  # '10' → Error on data[1]
    qc.x(data[1])
with qc.if_test((syndrome, 0b11)):  # '11' → Error on data[2]
    qc.x(data[2])

#Step 6 Decoding
qc.cx(data[0], data[2])
qc.cx(data[0], data[1])
qc.measure(data[0], output[0])

#Circuit
simulator = AerSimulator()
result = simulator.run(qc, shots=10000).result()
counts = result.get_counts()

#Result
output_counts = {'0': 0, '1': 0}
for key in counts:
    # Extract the last bit (output[0])
    output_bit = key.split()[0] if ' ' in key else key[-1]
    output_counts[output_bit] += counts[key]

print("Output counts:", output_counts)
print("Syndrome counts:", result.get_counts(qc))

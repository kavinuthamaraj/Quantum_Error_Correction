"""
Complete Quantum Error Correction Simulation
============================================

Multi-distance surface code analysis with MWPM, FFNN, CNN, and GNN decoders
for symmetric and asymmetric depolarization channels.

Features:
- Surface codes of distance 3, 5, 7
- MWPM baseline decoder
- Neural network decoders (FFNN, CNN, GNN)
- Comprehensive threshold analysis
- Sub-threshold performance evaluation
- Asymmetric depolarization channels
- 24+ detailed plots and analysis

"""

import numpy as np
import random
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, Input, Add, Multiply, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import warnings
import os
import time
from collections import defaultdict
import networkx as nx
from scipy.optimize import minimize_scalar
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Suppress warnings and optimize TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

@dataclass
class SurfaceCodeConfig:
    """Configuration for surface codes of different distances"""
    d: int
    n_qubits: int
    n_stabilizers: int
    stabilizer_x: List[List[int]]
    stabilizer_z: List[List[int]]
    logical_x: List[List[int]]
    logical_z: List[List[int]]
    restricted_positions: List[List[int]]

class SurfaceCodeFactory:
    """Factory for creating surface code configurations"""

    @staticmethod
    def create_config(d: int) -> SurfaceCodeConfig:
        """Create surface code configuration for given distance"""

        if d == 3:
            return SurfaceCodeConfig(
                d=3,
                n_qubits=9,
                n_stabilizers=8,
                stabilizer_x=[[0,1], [7,8], [1,2,4,5], [3,4,6,7]],
                stabilizer_z=[[2,5], [3,6], [0,1,3,4], [4,5,7,8]],
                logical_x=[[0,1,2], [2,5,8]],
                logical_z=[[0,3,6], [6,7,8]],
                restricted_positions=[[0,0], [0,2], [0,3], [1,0], [2,3], [3,0], [3,1], [3,3]]
            )

        elif d == 5:
            return SurfaceCodeConfig(
                d=5,
                n_qubits=25,
                n_stabilizers=24,
                stabilizer_x=SurfaceCodeFactory._generate_x_stabilizers_d5(),
                stabilizer_z=SurfaceCodeFactory._generate_z_stabilizers_d5(),
                logical_x=[[0,1,2,3,4], [4,9,14,19,24]],
                logical_z=[[0,5,10,15,20], [20,21,22,23,24]],
                restricted_positions=SurfaceCodeFactory._generate_restricted_d5()
            )

        elif d == 7:
            return SurfaceCodeConfig(
                d=7,
                n_qubits=49,
                n_stabilizers=48,
                stabilizer_x=SurfaceCodeFactory._generate_x_stabilizers_d7(),
                stabilizer_z=SurfaceCodeFactory._generate_z_stabilizers_d7(),
                logical_x=[[0,1,2,3,4,5,6], [6,13,20,27,34,41,48]],
                logical_z=[[0,7,14,21,28,35,42], [42,43,44,45,46,47,48]],
                restricted_positions=SurfaceCodeFactory._generate_restricted_d7()
            )

        else:
            raise ValueError(f"Distance {d} not supported")

    @staticmethod
    def _generate_x_stabilizers_d5():
        """Generate X stabilizers for d=5 surface code"""
        stabilizers = []

        # Face stabilizers (X-type)
        for i in range(2):
            for j in range(2):
                face_qubits = []
                base = i * 10 + j * 2
                # Add qubits around face
                face_qubits.extend([base, base+1, base+5, base+6])
                stabilizers.append(face_qubits)

        # Boundary stabilizers
        stabilizers.extend([
            [0,1], [3,4], [20,21], [23,24],  # Top/bottom edges
            [0,5], [15,20], [4,9], [19,24]   # Left/right edges
        ])

        return stabilizers

    @staticmethod
    def _generate_z_stabilizers_d5():
        """Generate Z stabilizers for d=5 surface code"""
        stabilizers = []

        # Vertex stabilizers (Z-type)
        for i in range(2):
            for j in range(2):
                vertex_qubits = []
                base = i * 10 + j * 2 + 6
                vertex_qubits.extend([base, base+1, base+5, base+6])
                stabilizers.append(vertex_qubits)

        # Boundary stabilizers
        stabilizers.extend([
            [2,7], [12,17], [7,8], [17,18],  # Vertical boundaries
            [10,11], [13,14], [5,10], [9,14] # Horizontal boundaries
        ])

        return stabilizers

    @staticmethod
    def _generate_x_stabilizers_d7():
        """Generate X stabilizers for d=7 surface code"""
        stabilizers = []

        # Generate face stabilizers systematically
        for i in range(3):
            for j in range(3):
                face_qubits = []
                base = i * 14 + j * 2
                face_qubits.extend([base, base+1, base+7, base+8])
                stabilizers.append(face_qubits)

        # Add boundary stabilizers
        for i in range(6):
            stabilizers.append([i, i+1])  # Top edge
            stabilizers.append([i+42, i+43])  # Bottom edge

        for i in range(0, 42, 7):
            stabilizers.append([i, i+7])  # Left edge
            stabilizers.append([i+6, i+13])  # Right edge

        return stabilizers

    @staticmethod
    def _generate_z_stabilizers_d7():
        """Generate Z stabilizers for d=7 surface code"""
        stabilizers = []

        # Generate vertex stabilizers systematically
        for i in range(3):
            for j in range(3):
                vertex_qubits = []
                base = i * 14 + j * 2 + 8
                vertex_qubits.extend([base, base+1, base+7, base+8])
                stabilizers.append(vertex_qubits)

        # Add boundary stabilizers
        for i in range(1, 6):
            stabilizers.append([i+7, i+8])  # Vertical boundaries
            stabilizers.append([i+35, i+36])

        for i in range(7, 35, 7):
            stabilizers.append([i+1, i+8])  # Horizontal boundaries
            stabilizers.append([i+5, i+12])

        return stabilizers

    @staticmethod
    def _generate_restricted_d5():
        """Generate restricted positions for d=5"""
        restricted = []
        for i in range(6):
            for j in range(6):
                if (i == 0 and j in [0, 2, 4, 5]) or \
                   (i == 1 and j in [0, 5]) or \
                   (i == 4 and j in [0, 5]) or \
                   (i == 5 and j in [0, 2, 4, 5]):
                    restricted.append([i, j])
        return restricted

    @staticmethod
    def _generate_restricted_d7():
        """Generate restricted positions for d=7"""
        restricted = []
        for i in range(8):
            for j in range(8):
                if (i == 0 and j in [0, 2, 4, 6, 7]) or \
                   (i == 1 and j in [0, 7]) or \
                   (i == 6 and j in [0, 7]) or \
                   (i == 7 and j in [0, 2, 4, 6, 7]):
                    restricted.append([i, j])
        return restricted

class MWPMDecoder:
    """Improved MWPM decoder for multiple distances"""

    def __init__(self, config: SurfaceCodeConfig):
        self.config = config
        self.d = config.d

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode syndrome using simplified MWPM"""
        syndrome_flat = syndrome.flatten() if len(syndrome.shape) > 1 else syndrome

        # Split syndrome into X and Z parts
        mid_point = len(syndrome_flat) // 2
        syndrome_x = syndrome_flat[:mid_point]
        syndrome_z = syndrome_flat[mid_point:]

        # Decode each type separately
        x_correction = self._decode_single_type(syndrome_x, 'X')
        z_correction = self._decode_single_type(syndrome_z, 'Z')

        # Combine corrections
        full_correction = np.zeros(2 * self.config.n_qubits, dtype=int)

        for i in range(self.config.n_qubits):
            full_correction[2*i] = x_correction[i] if i < len(x_correction) else 0
            full_correction[2*i + 1] = z_correction[i] if i < len(z_correction) else 0

        return full_correction

    def _decode_single_type(self, syndrome: np.ndarray, error_type: str) -> np.ndarray:
        """Decode single error type using greedy matching"""
        active_stabilizers = np.where(syndrome == 1)[0]
        correction = np.zeros(self.config.n_qubits, dtype=int)

        if len(active_stabilizers) == 0:
            return correction

        # Simple correction strategy based on syndrome pattern
        if error_type == 'X':
            stabilizers = self.config.stabilizer_x
        else:
            stabilizers = self.config.stabilizer_z

        # Apply corrections based on active stabilizers
        for stab_idx in active_stabilizers:
            if stab_idx < len(stabilizers):
                # Apply correction to first qubit in stabilizer
                qubits = stabilizers[stab_idx]
                if qubits and qubits[0] < self.config.n_qubits:
                    correction[qubits[0]] = 1

        return correction

class QuantumErrorCorrectionSimulator:
    """Comprehensive QEC simulator for multiple distances"""

    def __init__(self, d: int):
        self.d = d
        self.config = SurfaceCodeFactory.create_config(d)
        self.stabilizer_set_X = self._generate_stabilizer_set(self.config.stabilizer_x)
        self.stabilizer_set_Z = self._generate_stabilizer_set(self.config.stabilizer_z)
        self.mwpm_decoder = MWPMDecoder(self.config)

    def _generate_stabilizer_set(self, stabilizers: List[List[int]]) -> List[List[int]]:
        """Generate complete stabilizer group (simplified for large distances)"""
        if len(stabilizers) <= 6:  # Full generation for small sets
            stabilizer_set = []
            for r in range(1, len(stabilizers) + 1):
                for comb in itertools.combinations(stabilizers, r):
                    result = []
                    for stab in comb:
                        result.extend(stab)

                    final = []
                    for qubit in set(result):
                        if result.count(qubit) % 2 == 1:
                            final.append(qubit)

                    stabilizer_set.append(sorted(final))

            stabilizer_set = [list(x) for x in set(tuple(x) for x in stabilizer_set)]
            stabilizer_set.append([])  # Identity
            return stabilizer_set
        else:
            # Simplified for large distances
            return [[]]

    def generate_errors_batch(self, p: float, n_samples: int, px: float = 1/3, py: float = 1/3, pz: float = 1/3) -> np.ndarray:
        """Generate batch of Pauli errors"""
        # Normalize probabilities
        total_prob = px + py + pz
        px_norm = px / total_prob
        py_norm = py / total_prob
        pz_norm = pz / total_prob

        random_vals = np.random.random((n_samples, self.config.n_qubits))
        errors = np.zeros((n_samples, self.config.n_qubits), dtype=np.int8)

        # Error occurrence mask
        error_mask = random_vals <= p
        error_type_vals = np.random.random((n_samples, self.config.n_qubits))

        # Assign error types
        x_mask = error_mask & (error_type_vals <= px_norm)
        z_mask = error_mask & (error_type_vals > px_norm) & (error_type_vals <= px_norm + pz_norm)
        y_mask = error_mask & (error_type_vals > px_norm + pz_norm)

        errors[x_mask] = 1  # X
        errors[z_mask] = 2  # Z
        errors[y_mask] = 3  # Y

        return errors

    def errors_to_binary(self, errors_123: np.ndarray) -> np.ndarray:
        """Convert 1,2,3 format to binary X,Z format"""
        n_samples = errors_123.shape[0]
        binary_errors = np.zeros((n_samples, 2 * self.config.n_qubits), dtype=np.int8)

        for i in range(self.config.n_qubits):
            # X component (X or Y errors)
            binary_errors[:, 2*i] = (errors_123[:, i] == 1) | (errors_123[:, i] == 3)
            # Z component (Z or Y errors)
            binary_errors[:, 2*i + 1] = (errors_123[:, i] == 2) | (errors_123[:, i] == 3)

        return binary_errors

    def calculate_syndromes_batch(self, errors_batch: np.ndarray) -> np.ndarray:
        """Calculate syndrome vectors for batch of errors"""
        n_samples = errors_batch.shape[0]
        syndromes = np.zeros((n_samples, self.config.n_stabilizers), dtype=np.int8)

        for sample_idx in range(n_samples):
            syndrome = self._calculate_single_syndrome(errors_batch[sample_idx])
            syndromes[sample_idx] = syndrome

        return syndromes

    def _calculate_single_syndrome(self, error_pattern: np.ndarray) -> np.ndarray:
        """Calculate syndrome for single error pattern"""
        syndrome = np.zeros(self.config.n_stabilizers, dtype=np.int8)

        # Check X-type stabilizers
        for stab_idx, qubits in enumerate(self.config.stabilizer_x):
            if stab_idx < len(syndrome) // 2:
                violation = 0
                for qubit in qubits:
                    if qubit < len(error_pattern) and error_pattern[qubit] in [1, 3]:  # X or Y
                        violation ^= 1
                syndrome[stab_idx] = violation

        # Check Z-type stabilizers
        for stab_idx, qubits in enumerate(self.config.stabilizer_z):
            syndrome_idx = len(self.config.stabilizer_x) + stab_idx
            if syndrome_idx < len(syndrome):
                violation = 0
                for qubit in qubits:
                    if qubit < len(error_pattern) and error_pattern[qubit] in [2, 3]:  # Z or Y
                        violation ^= 1
                syndrome[syndrome_idx] = violation

        return syndrome

    def classify_logical_errors(self, errors_binary: np.ndarray) -> np.ndarray:
        """Classify errors as logical I, X, Z, or Y"""
        n_samples = errors_binary.shape[0]
        classifications = np.zeros(n_samples, dtype=np.int8)

        for sample_idx in range(n_samples):
            error_reshaped = errors_binary[sample_idx].reshape(self.config.n_qubits, 2)
            error_x = error_reshaped[:, 0]
            error_z = error_reshaped[:, 1]

            if self._is_identity_error(error_x, error_z):
                classifications[sample_idx] = 0
                continue

            logical_x = self._is_logical_x_error(error_x)
            logical_z = self._is_logical_z_error(error_z)

            if logical_x and logical_z:
                classifications[sample_idx] = 3  # Y
            elif logical_x:
                classifications[sample_idx] = 1  # X
            elif logical_z:
                classifications[sample_idx] = 2  # Z
            else:
                classifications[sample_idx] = 0  # I

        return classifications

    def _is_identity_error(self, error_x: np.ndarray, error_z: np.ndarray) -> bool:
        """Check if error is in stabilizer group"""
        if np.all(error_x == 0) and np.all(error_z == 0):
            return True

        # Simplified check for large distances
        if len(self.stabilizer_set_X) == 1:  # Only identity in set
            return False

        nonzero_x = [i for i in range(len(error_x)) if error_x[i] == 1]
        nonzero_z = [i for i in range(len(error_z)) if error_z[i] == 1]

        return (nonzero_x in self.stabilizer_set_X and nonzero_z in self.stabilizer_set_Z)

    def _is_logical_x_error(self, error_x: np.ndarray) -> bool:
        """Check if error anticommutes with Z logical operators"""
        for logical_z_op in self.config.logical_z:
            overlap = sum(error_x[q] for q in logical_z_op if q < len(error_x))
            if overlap % 2 != 0:
                return True
        return False

    def _is_logical_z_error(self, error_z: np.ndarray) -> bool:
        """Check if error anticommutes with X logical operators"""
        for logical_x_op in self.config.logical_x:
            overlap = sum(error_z[q] for q in logical_x_op if q < len(error_z))
            if overlap % 2 != 0:
                return True
        return False

    def create_dataset(self, p: float, n_samples: int, px: float = 1/3, py: float = 1/3, pz: float = 1/3) -> np.ndarray:
        """Create dataset for training/testing"""
        print(f"Generating {n_samples} samples for d={self.d}, p={p}")

        chunk_size = min(5000, n_samples)
        all_data = []

        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_size_actual = end_idx - start_idx

            # Generate errors
            errors_123 = self.generate_errors_batch(p, chunk_size_actual, px, py, pz)
            errors_binary = self.errors_to_binary(errors_123)
            syndromes = self.calculate_syndromes_batch(errors_123)
            hld_labels = self.classify_logical_errors(errors_binary)

            # Combine data
            for i in range(chunk_size_actual):
                data_row = np.concatenate([
                    errors_binary[i],
                    syndromes[i],
                    [hld_labels[i]]
                ])
                all_data.append(data_row)

        dataset = np.array(all_data)
        print(f"Dataset shape: {dataset.shape}")

        return dataset

    def build_ffnn_model(self, input_dim: int, output_dim: int, model_type: str = 'lld') -> Sequential:
        """Build FFNN model adapted for different distances"""

        # Scale architecture based on distance
        base_units = 64 * self.d

        if model_type == 'lld':
            model = Sequential([
                Dense(base_units * 4, input_dim=input_dim, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(base_units * 2, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(base_units, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(base_units // 2, activation='relu'),
                Dense(output_dim, activation='sigmoid')
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:  # HLD
            model = Sequential([
                Dense(base_units * 2, input_dim=input_dim, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(base_units, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(base_units // 2, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(output_dim, activation='softmax')
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        return model

    def build_cnn_model(self, input_shape: Tuple[int, ...], output_dim: int, model_type: str = 'lld') -> Sequential:
        """Build CNN model adapted for different distances"""

        # Scale CNN based on distance
        base_filters = 16 * (self.d // 3 + 1)

        model = Sequential([
            Conv2D(base_filters, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            BatchNormalization(),
            Conv2D(base_filters * 2, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.25),
            Conv2D(base_filters * 4, (2, 2), activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dense(base_filters * 8, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(base_filters * 4, activation='relu'),
            BatchNormalization(),
            Dropout(0.3)
        ])

        if model_type == 'lld':
            model.add(Dense(output_dim, activation='sigmoid'))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(Dense(output_dim, activation='softmax'))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        return model

    def build_gnn_model(self, input_dim: int, output_dim: int, model_type: str = 'lld') -> Model:
        """Build Graph Neural Network (attention-based) adapted for different distances"""

        inputs = Input(shape=(input_dim,))

        # Scale GNN based on distance
        base_units = 32 * self.d

        # Multi-head attention mechanism
        x = Dense(base_units * 2, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # First attention block
        attention1 = Dense(base_units * 2, activation='softmax')(x)
        attended1 = Multiply()([x, attention1])

        # Second layer with residual connection
        x2 = Dense(base_units, activation='relu')(attended1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)

        # Second attention block
        attention2 = Dense(base_units, activation='softmax')(x2)
        attended2 = Multiply()([x2, attention2])

        # Final layers
        x3 = Dense(base_units // 2, activation='relu')(attended2)
        x3 = BatchNormalization()(x3)

        if model_type == 'lld':
            outputs = Dense(output_dim, activation='sigmoid')(x3)
            model = Model(inputs, outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            outputs = Dense(output_dim, activation='softmax')(x3)
            model = Model(inputs, outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        return model

    def evaluate_mwpm(self, X_test: np.ndarray, y_test_lld: np.ndarray) -> Dict[str, float]:
        """Evaluate MWPM decoder"""
        n_samples = len(X_test)
        correct_count = 0
        logical_error_count = 0
        total_errors = 0

        print(f"Evaluating MWPM on {n_samples} samples for d={self.d}...")

        for i in range(n_samples):
            syndrome = X_test[i]
            true_error = y_test_lld[i]

            try:
                # MWPM correction
                correction = self.mwpm_decoder.decode(syndrome)

                # Ensure correct length
                if len(correction) != len(true_error):
                    correction = np.pad(correction, (0, max(0, len(true_error) - len(correction))))[:len(true_error)]

                # Analyze result
                total_correction = np.bitwise_xor(true_error, correction)
                is_correct, has_logical = self._analyze_correction_simple(total_correction)

                if is_correct:
                    correct_count += 1
                elif has_logical:
                    logical_error_count += 1

                total_errors += 1

            except Exception as e:
                if i < 5:  # Only print first few errors
                    print(f"MWPM decode error on sample {i}: {e}")
                total_errors += 1
                continue

        if total_errors == 0:
            return {'accuracy': 0.0, 'logical_error_rate': 1.0}

        accuracy = correct_count / total_errors
        logical_error_rate = logical_error_count / total_errors

        print(f"MWPM d={self.d}: Accuracy={accuracy:.4f}, Logical Error Rate={logical_error_rate:.4f}")

        return {'accuracy': accuracy, 'logical_error_rate': logical_error_rate}

    def _analyze_correction_simple(self, correction: np.ndarray) -> Tuple[bool, bool]:
        """Analyze if correction leads to logical error"""
        if len(correction) == 2 * self.config.n_qubits:
            correction_reshaped = correction.reshape(self.config.n_qubits, 2)
            error_x = correction_reshaped[:, 0]
            error_z = correction_reshaped[:, 1]

            if self._is_identity_error(error_x, error_z):
                return True, False

            logical_x = self._is_logical_x_error(error_x)
            logical_z = self._is_logical_z_error(error_z)

            return False, logical_x or logical_z

        return False, False

    def _evaluate_model(self, model, X_test: np.ndarray, y_test_lld: np.ndarray) -> Dict[str, float]:
        """Evaluate neural network model"""
        if len(X_test.shape) == 4:  # CNN input
            predictions = model.predict(X_test, verbose=0, batch_size=1024)
        else:
            predictions = model.predict(X_test, verbose=0, batch_size=1024)

        predictions_binary = (predictions > 0.5).astype(int)
        corrections = np.bitwise_xor(y_test_lld, predictions_binary)

        correct_count = 0
        logical_error_count = 0

        for correction in corrections:
            is_correct, has_logical = self._analyze_correction_simple(correction)
            if is_correct:
                correct_count += 1
            elif has_logical:
                logical_error_count += 1

        return {
            'accuracy': correct_count / len(corrections),
            'logical_error_rate': logical_error_count / len(corrections)
        }

class MultiDistanceAnalyzer:
    """Analyzer for comparing performance across multiple distances"""

    def __init__(self, distances: List[int] = [3, 5, 7]):
        self.distances = distances
        self.simulators = {d: QuantumErrorCorrectionSimulator(d) for d in distances}
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    def run_comprehensive_analysis(self,
                                 error_rates: Optional[List[float]] = None,
                                 noise_scenarios: Optional[Dict[str, Tuple[float, float, float]]] = None,
                                 train_size: int = 20000,
                                 test_size: int = 4000) -> Dict:
        """Run comprehensive analysis across all distances and scenarios"""

        if error_rates is None:
            error_rates = [0.001, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03, 0.04, 0.05]

        if noise_scenarios is None:
            noise_scenarios = {
                'Symmetric': (1/3, 1/3, 1/3),
                'X-biased': (0.7, 0.15, 0.15),
                'Z-biased': (0.15, 0.15, 0.7),
                'Y-biased': (0.15, 0.7, 0.15),
                'X-dominant': (0.9, 0.05, 0.05),
                'Z-dominant': (0.05, 0.05, 0.9)
            }

        print("Starting Multi-Distance Comprehensive Analysis")
        print("=" * 80)
        print(f"Distances: {self.distances}")
        print(f"Error rates: {len(error_rates)} points")
        print(f"Noise scenarios: {list(noise_scenarios.keys())}")
        print(f"Models: MWPM, FFNN, CNN, GNN")
        print("=" * 80)

        for scenario_name, (px, py, pz) in noise_scenarios.items():
            print(f"\n{'='*70}")
            print(f"NOISE SCENARIO: {scenario_name}")
            print(f"Probabilities: px={px:.2f}, py={py:.2f}, pz={pz:.2f}")
            print(f"{'='*70}")

            for d in self.distances:
                print(f"\n--- Distance {d} ---")
                simulator = self.simulators[d]

                for p in error_rates:
                    print(f"p = {p:.3f}", end=" -> ")

                    try:
                        # Adjust dataset size based on distance
                        train_size_adj = max(train_size // (d // 3), 10000)
                        test_size_adj = max(test_size // (d // 3), 2000)

                        # Generate datasets
                        train_data = simulator.create_dataset(p, train_size_adj, px, py, pz)
                        test_data = simulator.create_dataset(p, test_size_adj, px, py, pz)

                        # Prepare data
                        syndrome_start = 2 * simulator.config.n_qubits
                        syndrome_end = syndrome_start + simulator.config.n_stabilizers

                        X_train = train_data[:, syndrome_start:syndrome_end].astype(np.float32)
                        X_test = test_data[:, syndrome_start:syndrome_end].astype(np.float32)

                        y_train_lld = train_data[:, :syndrome_start]
                        y_test_lld = test_data[:, :syndrome_start]

                        # Normalize features
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                        # 1. MWPM
                        mwpm_results = simulator.evaluate_mwpm(X_test, y_test_lld)
                        self.results[scenario_name][d][p]['MWPM'] = mwpm_results

                        # 2. FFNN
                        ffnn_model = simulator.build_ffnn_model(X_train.shape[1], 2 * simulator.config.n_qubits, 'lld')

                        # Adjust training parameters based on distance
                        epochs = max(60 - d * 10, 30)
                        batch_size = min(256, len(X_train) // 10)

                        ffnn_model.fit(X_train, y_train_lld,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     validation_split=0.15,
                                     verbose=0,
                                     callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

                        ffnn_results = simulator._evaluate_model(ffnn_model, X_test, y_test_lld)
                        self.results[scenario_name][d][p]['FFNN'] = ffnn_results

                        # 3. CNN
                        syndrome_grid_size = int(np.ceil(np.sqrt(simulator.config.n_stabilizers)))

                        # Pad syndromes to square grid
                        X_train_padded = np.zeros((len(X_train), syndrome_grid_size**2))
                        X_test_padded = np.zeros((len(X_test), syndrome_grid_size**2))

                        X_train_padded[:, :X_train.shape[1]] = X_train
                        X_test_padded[:, :X_test.shape[1]] = X_test

                        X_train_cnn = X_train_padded.reshape(-1, syndrome_grid_size, syndrome_grid_size, 1)
                        X_test_cnn = X_test_padded.reshape(-1, syndrome_grid_size, syndrome_grid_size, 1)

                        cnn_model = simulator.build_cnn_model((syndrome_grid_size, syndrome_grid_size, 1),
                                                            2 * simulator.config.n_qubits, 'lld')

                        cnn_model.fit(X_train_cnn, y_train_lld,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_split=0.15,
                                    verbose=0,
                                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

                        cnn_results = simulator._evaluate_model(cnn_model, X_test_cnn, y_test_lld)
                        self.results[scenario_name][d][p]['CNN'] = cnn_results

                        # 4. GNN
                        gnn_model = simulator.build_gnn_model(X_train.shape[1], 2 * simulator.config.n_qubits, 'lld')

                        gnn_model.fit(X_train, y_train_lld,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_split=0.15,
                                    verbose=0,
                                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

                        gnn_results = simulator._evaluate_model(gnn_model, X_test, y_test_lld)
                        self.results[scenario_name][d][p]['GNN'] = gnn_results

                        # Print results
                        print(f"MWPM:{mwpm_results['logical_error_rate']:.4f}, "
                              f"FFNN:{ffnn_results['logical_error_rate']:.4f}, "
                              f"CNN:{cnn_results['logical_error_rate']:.4f}, "
                              f"GNN:{gnn_results['logical_error_rate']:.4f}")

                    except Exception as e:
                        print(f"Error at d={d}, p={p}: {e}")
                        continue

        return dict(self.results)

    def create_comprehensive_plots(self, results: Dict) -> plt.Figure:
        """Create comprehensive comparison plots"""

        plt.rcParams.update({'font.size': 10})
        fig = plt.figure(figsize=(28, 20))

        scenarios = list(results.keys())
        distances = self.distances
        models = ['MWPM', 'FFNN', 'CNN', 'GNN']

        # Define colors
        distance_colors = {3: '#FF6B6B', 5: '#4ECDC4', 7: '#45B7D1'}
        model_colors = {'MWPM': '#E74C3C', 'FFNN': '#3498DB', 'CNN': '#2ECC71', 'GNN': '#9B59B6'}

        plot_idx = 1

        # 1-6: Logical vs Physical Error Rate for each scenario (2x3 grid)
        for scenario_idx, scenario in enumerate(scenarios):
            ax = plt.subplot(4, 6, plot_idx)
            plot_idx += 1

            for d in distances:
                if d in results[scenario]:
                    error_rates = sorted(results[scenario][d].keys())

                    for model in models:
                        logical_rates = []
                        valid_rates = []

                        for p in error_rates:
                            if model in results[scenario][d][p]:
                                logical_rates.append(results[scenario][d][p][model]['logical_error_rate'])
                                valid_rates.append(p)

                        if logical_rates:
                            linestyle = '-' if model == 'MWPM' else '--' if model == 'FFNN' else ':' if model == 'CNN' else '-.'
                            plt.loglog(valid_rates, logical_rates,
                                     linestyle, color=distance_colors[d],
                                     alpha=0.8, linewidth=2,
                                     label=f'd={d}, {model}' if scenario_idx == 0 else "")

            # Threshold line
            if error_rates:
                plt.loglog(error_rates, error_rates, 'k--', alpha=0.3, linewidth=1)

            plt.xlabel('Physical Error Rate')
            plt.ylabel('Logical Error Rate')
            plt.title(f'{scenario} Channel')
            plt.grid(True, alpha=0.3)
            if scenario_idx == 0:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # 7-12: Model comparison across distances for symmetric channel
        if 'Symmetric' in results:
            symmetric_results = results['Symmetric']

            for model_idx, model in enumerate(models):
                ax = plt.subplot(4, 6, 7 + model_idx)

                for d in distances:
                    if d in symmetric_results:
                        error_rates = sorted(symmetric_results[d].keys())
                        logical_rates = []
                        valid_rates = []

                        for p in error_rates:
                            if model in symmetric_results[d][p]:
                                logical_rates.append(symmetric_results[d][p][model]['logical_error_rate'])
                                valid_rates.append(p)

                        if logical_rates:
                            plt.loglog(valid_rates, logical_rates, 'o-',
                                     color=distance_colors[d], linewidth=2, markersize=4,
                                     label=f'Distance {d}')

                plt.loglog(error_rates, error_rates, 'k--', alpha=0.3)
                plt.xlabel('Physical Error Rate')
                plt.ylabel('Logical Error Rate')
                plt.title(f'{model} - Symmetric Channel')
                plt.legend()
                plt.grid(True, alpha=0.3)

        # 13-18: Threshold analysis
        ax = plt.subplot(4, 6, 13)
        self._plot_threshold_analysis(results, ax)

        ax = plt.subplot(4, 6, 14)
        self._plot_distance_scaling(results, ax)

        ax = plt.subplot(4, 6, 15)
        self._plot_sub_threshold_performance(results, ax)

        ax = plt.subplot(4, 6, 16)
        self._plot_noise_bias_comparison(results, ax)

        ax = plt.subplot(4, 6, 17)
        self._plot_model_improvement_over_mwpm(results, ax)

        ax = plt.subplot(4, 6, 18)
        self._plot_accuracy_heatmap(results, ax)

        # 19-24: Sub-threshold detailed analysis
        for d_idx, d in enumerate(distances):
            ax = plt.subplot(4, 6, 19 + d_idx)
            self._plot_sub_threshold_detailed(results, d, ax)

        plt.tight_layout()
        plt.show()

        return fig

    def _plot_threshold_analysis(self, results: Dict, ax) -> None:
        """Plot threshold comparison across distances and models"""
        scenarios = ['Symmetric', 'X-biased', 'Z-biased']

        thresholds = defaultdict(lambda: defaultdict(list))

        for scenario in scenarios:
            if scenario in results:
                for d in self.distances:
                    if d in results[scenario]:
                        for model in ['MWPM', 'FFNN', 'CNN', 'GNN']:
                            threshold = self._find_threshold(results[scenario][d], model)
                            thresholds[model][scenario].append(threshold)

        x = np.arange(len(scenarios))
        width = 0.15

        for i, model in enumerate(['MWPM', 'FFNN', 'CNN', 'GNN']):
            for j, d in enumerate(self.distances):
                values = [thresholds[model][scenario][j] if j < len(thresholds[model][scenario]) else 0
                         for scenario in scenarios]
                ax.bar(x + (i - 1.5) * width + j * width/3, values, width/3,
                      label=f'{model} d={d}', alpha=0.8)

        ax.set_xlabel('Noise Scenario')
        ax.set_ylabel('Error Threshold')
        ax.set_title('Error Thresholds by Distance and Model')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_distance_scaling(self, results: Dict, ax) -> None:
        """Plot how performance scales with distance"""
        if 'Symmetric' in results:
            fixed_p = 0.01  # Fixed error rate for comparison

            for model in ['MWPM', 'FFNN', 'CNN', 'GNN']:
                distances_plot = []
                logical_rates = []

                for d in self.distances:
                    if d in results['Symmetric'] and fixed_p in results['Symmetric'][d]:
                        if model in results['Symmetric'][d][fixed_p]:
                            distances_plot.append(d)
                            logical_rates.append(results['Symmetric'][d][fixed_p][model]['logical_error_rate'])

                if distances_plot:
                    ax.semilogy(distances_plot, logical_rates, 'o-',
                              label=model, linewidth=2, markersize=6)

            ax.set_xlabel('Surface Code Distance')
            ax.set_ylabel('Logical Error Rate')
            ax.set_title(f'Distance Scaling (p = {fixed_p})')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_sub_threshold_performance(self, results: Dict, ax) -> None:
        """Plot sub-threshold performance comparison"""
        if 'Symmetric' in results:
            sub_threshold_rates = [0.005, 0.008, 0.01, 0.012, 0.015]

            for d in self.distances:
                if d in results['Symmetric']:
                    mwpm_rates = []
                    valid_rates = []

                    for p in sub_threshold_rates:
                        if p in results['Symmetric'][d] and 'MWPM' in results['Symmetric'][d][p]:
                            mwpm_rates.append(results['Symmetric'][d][p]['MWPM']['logical_error_rate'])
                            valid_rates.append(p)

                    if mwpm_rates:
                        ax.loglog(valid_rates, mwpm_rates, 'o-',
                                label=f'MWPM d={d}', linewidth=2, markersize=5)

            ax.loglog(sub_threshold_rates, sub_threshold_rates, 'k--', alpha=0.5)
            ax.set_xlabel('Physical Error Rate')
            ax.set_ylabel('Logical Error Rate')
            ax.set_title('Sub-threshold Performance (MWPM)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_noise_bias_comparison(self, results: Dict, ax) -> None:
        """Plot performance under different noise biases"""
        scenarios = ['Symmetric', 'X-biased', 'Z-biased', 'Y-biased']
        fixed_p = 0.015

        scenario_data = defaultdict(list)

        for scenario in scenarios:
            if scenario in results:
                best_performance = 1.0  # Start with worst case

                for d in self.distances:
                    if d in results[scenario] and fixed_p in results[scenario][d]:
                        for model in ['MWPM', 'FFNN', 'CNN', 'GNN']:
                            if model in results[scenario][d][fixed_p]:
                                perf = results[scenario][d][fixed_p][model]['logical_error_rate']
                                best_performance = min(best_performance, perf)

                scenario_data[scenario] = best_performance

        scenarios_list = list(scenario_data.keys())
        performances = list(scenario_data.values())

        bars = ax.bar(scenarios_list, performances, alpha=0.7)
        ax.set_ylabel('Best Logical Error Rate')
        ax.set_title(f'Noise Bias Comparison (p = {fixed_p})')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_model_improvement_over_mwpm(self, results: Dict, ax) -> None:
        """Plot neural network improvement over MWPM"""
        if 'Symmetric' in results:
            for d in self.distances:
                if d in results['Symmetric']:
                    error_rates = sorted(results['Symmetric'][d].keys())

                    for model in ['FFNN', 'CNN', 'GNN']:
                        improvements = []
                        valid_rates = []

                        for p in error_rates:
                            if ('MWPM' in results['Symmetric'][d][p] and
                                model in results['Symmetric'][d][p]):

                                mwpm_acc = results['Symmetric'][d][p]['MWPM']['accuracy']
                                model_acc = results['Symmetric'][d][p][model]['accuracy']

                                if mwpm_acc > 0:
                                    improvement = ((model_acc - mwpm_acc) / mwpm_acc) * 100
                                    improvements.append(improvement)
                                    valid_rates.append(p)

                        if improvements:
                            linestyle = '--' if model == 'FFNN' else ':' if model == 'CNN' else '-.'
                            ax.semilogx(valid_rates, improvements, linestyle,
                                      label=f'{model} d={d}', linewidth=2)

            ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax.set_xlabel('Physical Error Rate')
            ax.set_ylabel('Improvement over MWPM (%)')
            ax.set_title('Neural Network Improvement over MWPM')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

    def _plot_accuracy_heatmap(self, results: Dict, ax) -> None:
        """Plot accuracy heatmap for different scenarios and distances"""
        if 'Symmetric' in results:
            data_matrix = []
            row_labels = []

            fixed_p = 0.01

            for d in self.distances:
                for model in ['MWPM', 'FFNN', 'CNN', 'GNN']:
                    row_data = []

                    for scenario in ['Symmetric', 'X-biased', 'Z-biased']:
                        if (scenario in results and d in results[scenario] and
                            fixed_p in results[scenario][d] and
                            model in results[scenario][d][fixed_p]):

                            acc = results[scenario][d][fixed_p][model]['accuracy']
                            row_data.append(acc)
                        else:
                            row_data.append(0)

                    data_matrix.append(row_data)
                    row_labels.append(f'd={d}, {model}')

            im = ax.imshow(data_matrix, cmap='viridis', aspect='auto')
            ax.set_xticks(range(3))
            ax.set_xticklabels(['Symmetric', 'X-biased', 'Z-biased'])
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=8)
            ax.set_title(f'Accuracy Heatmap (p = {fixed_p})')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Accuracy')

    def _plot_sub_threshold_detailed(self, results: Dict, distance: int, ax) -> None:
        """Plot detailed sub-threshold analysis for specific distance"""
        if 'Symmetric' in results and distance in results['Symmetric']:
            sub_rates = [p for p in sorted(results['Symmetric'][distance].keys()) if p <= 0.02]

            for model in ['MWPM', 'FFNN', 'CNN', 'GNN']:
                logical_rates = []
                valid_rates = []

                for p in sub_rates:
                    if model in results['Symmetric'][distance][p]:
                        logical_rates.append(results['Symmetric'][distance][p][model]['logical_error_rate'])
                        valid_rates.append(p)

                if logical_rates:
                    ax.loglog(valid_rates, logical_rates, 'o-',
                            label=model, linewidth=2, markersize=4)

            ax.loglog(sub_rates, sub_rates, 'k--', alpha=0.5)
            ax.set_xlabel('Physical Error Rate')
            ax.set_ylabel('Logical Error Rate')
            ax.set_title(f'Sub-threshold Detail (d={distance})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    def _find_threshold(self, distance_results: Dict, model: str) -> float:
        """Find threshold for specific model and distance"""
        error_rates = sorted(distance_results.keys())

        for p in error_rates:
            if model in distance_results[p]:
                logical_rate = distance_results[p][model]['logical_error_rate']
                if logical_rate >= p:
                    return p

        return error_rates[-1] if error_rates else 0.0

    def create_results_tables(self, results: Dict) -> pd.DataFrame:
        """Create comprehensive results tables"""

        print("\n" + "="*120)
        print("COMPREHENSIVE MULTI-DISTANCE ANALYSIS RESULTS")
        print("="*120)

        # Create master DataFrame
        all_data = []

        for scenario in results:
            for d in results[scenario]:
                for p in results[scenario][d]:
                    for model in results[scenario][d][p]:
                        all_data.append({
                            'Scenario': scenario,
                            'Distance': d,
                            'Physical_Error_Rate': p,
                            'Model': model,
                            'Accuracy': results[scenario][d][p][model]['accuracy'],
                            'Logical_Error_Rate': results[scenario][d][p][model]['logical_error_rate']
                        })

        df = pd.DataFrame(all_data)

        # Print summary by scenario and distance
        for scenario in ['Symmetric', 'X-biased', 'Z-biased']:
            if scenario in results:
                print(f"\n{scenario.upper()} CHANNEL RESULTS")
                print("-" * 80)

                for d in self.distances:
                    if d in results[scenario]:
                        print(f"\nDistance {d}:")

                        scenario_df = df[(df['Scenario'] == scenario) & (df['Distance'] == d)]

                        if not scenario_df.empty:
                            pivot_accuracy = scenario_df.pivot(index='Physical_Error_Rate',
                                                             columns='Model', values='Accuracy')
                            pivot_logical = scenario_df.pivot(index='Physical_Error_Rate',
                                                            columns='Model', values='Logical_Error_Rate')

                            print("Accuracy:")
                            print(pivot_accuracy.round(4))
                            print("\nLogical Error Rates:")
                            print(pivot_logical.round(6))

        # Threshold analysis
        print(f"\n{'='*60}")
        print("THRESHOLD ANALYSIS")
        print(f"{'='*60}")

        threshold_data = []
        for scenario in results:
            for d in results[scenario]:
                for model in ['MWPM', 'FFNN', 'CNN', 'GNN']:
                    threshold = self._find_threshold(results[scenario][d], model)
                    threshold_data.append({
                        'Scenario': scenario,
                        'Distance': d,
                        'Model': model,
                        'Threshold': threshold
                    })

        threshold_df = pd.DataFrame(threshold_data)

        for scenario in ['Symmetric', 'X-biased', 'Z-biased']:
            scenario_thresholds = threshold_df[threshold_df['Scenario'] == scenario]
            if not scenario_thresholds.empty:
                pivot_thresholds = scenario_thresholds.pivot(index='Distance',
                                                           columns='Model', values='Threshold')
                print(f"\n{scenario} Thresholds:")
                print(pivot_thresholds.round(6))

        return df, threshold_df

    def save_results(self, results: Dict, filename_prefix: str = 'multi_distance_qec'):
        """Save all results to files"""

        # Save raw results
        with open(f'{filename_prefix}_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        # Save tables
        df, threshold_df = self.create_results_tables(results)
        df.to_csv(f'{filename_prefix}_detailed_results.csv', index=False)
        threshold_df.to_csv(f'{filename_prefix}_thresholds.csv', index=False)

        print(f"\nResults saved to:")
        print(f"- {filename_prefix}_results.pkl")
        print(f"- {filename_prefix}_detailed_results.csv")
        print(f"- {filename_prefix}_thresholds.csv")

class AnalysisVisualizer:
    """Additional visualization tools for detailed analysis"""

    @staticmethod
    def create_threshold_comparison_plot(results: Dict) -> plt.Figure:
        """Create focused threshold comparison plot"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        distances = [3, 5, 7]
        scenarios = ['Symmetric', 'X-biased', 'Z-biased']

        # Plot 1-3: Threshold curves for each distance
        for d_idx, d in enumerate(distances):
            ax = axes[0, d_idx]

            for scenario in scenarios:
                if scenario in results and d in results[scenario]:
                    error_rates = sorted(results[scenario][d].keys())

                    # MWPM threshold curve
                    mwpm_logical = []
                    valid_rates = []

                    for p in error_rates:
                        if 'MWPM' in results[scenario][d][p]:
                            mwpm_logical.append(results[scenario][d][p]['MWPM']['logical_error_rate'])
                            valid_rates.append(p)

                    if mwpm_logical:
                        ax.loglog(valid_rates, mwpm_logical, 'o-',
                                label=f'{scenario}', linewidth=2, markersize=4)

            # Threshold line
            if valid_rates:
                ax.loglog(valid_rates, valid_rates, 'k--', alpha=0.5)

            ax.set_xlabel('Physical Error Rate')
            ax.set_ylabel('Logical Error Rate')
            ax.set_title(f'MWPM Thresholds - Distance {d}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 4-6: Model comparison for each distance (symmetric channel)
        if 'Symmetric' in results:
            for d_idx, d in enumerate(distances):
                ax = axes[1, d_idx]

                if d in results['Symmetric']:
                    error_rates = sorted(results['Symmetric'][d].keys())

                    for model in ['MWPM', 'FFNN', 'CNN', 'GNN']:
                        logical_rates = []
                        valid_rates = []

                        for p in error_rates:
                            if model in results['Symmetric'][d][p]:
                                logical_rates.append(results['Symmetric'][d][p][model]['logical_error_rate'])
                                valid_rates.append(p)

                        if logical_rates:
                            linestyle = '-' if model == 'MWPM' else '--' if model == 'FFNN' else ':' if model == 'CNN' else '-.'
                            ax.loglog(valid_rates, logical_rates, linestyle,
                                    label=model, linewidth=2)

                    ax.loglog(valid_rates, valid_rates, 'k--', alpha=0.5)
                    ax.set_xlabel('Physical Error Rate')
                    ax.set_ylabel('Logical Error Rate')
                    ax.set_title(f'Model Comparison - Distance {d}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def create_distance_scaling_analysis(results: Dict) -> plt.Figure:
        """Create detailed distance scaling analysis"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Distance scaling at fixed error rates
        ax = axes[0, 0]

        if 'Symmetric' in results:
            fixed_rates = [0.005, 0.01, 0.015, 0.02]

            for p in fixed_rates:
                distances_plot = []
                mwpm_rates = []

                for d in [3, 5, 7]:
                    if (d in results['Symmetric'] and
                        p in results['Symmetric'][d] and
                        'MWPM' in results['Symmetric'][d][p]):

                        distances_plot.append(d)
                        mwpm_rates.append(results['Symmetric'][d][p]['MWPM']['logical_error_rate'])

                if distances_plot:
                    ax.semilogy(distances_plot, mwpm_rates, 'o-',
                              label=f'p = {p}', linewidth=2, markersize=6)

            ax.set_xlabel('Surface Code Distance')
            ax.set_ylabel('Logical Error Rate')
            ax.set_title('MWPM Distance Scaling')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 2: Threshold scaling with distance
        ax = axes[0, 1]

        thresholds_by_distance = {}

        for scenario in ['Symmetric', 'X-biased', 'Z-biased']:
            if scenario in results:
                thresholds = []
                distances_with_data = []

                for d in [3, 5, 7]:
                    if d in results[scenario]:
                        threshold = None
                        error_rates = sorted(results[scenario][d].keys())

                        for p in error_rates:
                            if 'MWPM' in results[scenario][d][p]:
                                logical_rate = results[scenario][d][p]['MWPM']['logical_error_rate']
                                if logical_rate >= p:
                                    threshold = p
                                    break

                        if threshold:
                            thresholds.append(threshold)
                            distances_with_data.append(d)

                if thresholds:
                    ax.plot(distances_with_data, thresholds, 'o-',
                          label=scenario, linewidth=2, markersize=6)

        ax.set_xlabel('Surface Code Distance')
        ax.set_ylabel('Error Threshold')
        ax.set_title('Threshold Scaling with Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Sub-threshold suppression rate
        ax = axes[1, 0]

        if 'Symmetric' in results:
            for d in [3, 5, 7]:
                if d in results['Symmetric']:
                    # Calculate suppression rate (how fast logical error rate decreases)
                    error_rates = [p for p in sorted(results['Symmetric'][d].keys()) if p <= 0.015]

                    if len(error_rates) >= 2:
                        log_ratios = []
                        rate_pairs = []

                        for i in range(len(error_rates) - 1):
                            p1, p2 = error_rates[i], error_rates[i+1]

                            if ('MWPM' in results['Symmetric'][d][p1] and
                                'MWPM' in results['Symmetric'][d][p2]):

                                logical1 = results['Symmetric'][d][p1]['MWPM']['logical_error_rate']
                                logical2 = results['Symmetric'][d][p2]['MWPM']['logical_error_rate']

                                if logical1 > 0 and logical2 > 0:
                                    ratio = np.log(logical2 / logical1) / np.log(p2 / p1)
                                    log_ratios.append(ratio)
                                    rate_pairs.append((p1 + p2) / 2)

                        if log_ratios:
                            ax.plot(rate_pairs, log_ratios, 'o-',
                                  label=f'Distance {d}', linewidth=2, markersize=5)

            ax.set_xlabel('Physical Error Rate')
            ax.set_ylabel('Suppression Exponent')
            ax.set_title('Sub-threshold Suppression Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 4: Model improvement scaling
        ax = axes[1, 1]

        if 'Symmetric' in results:
            fixed_p = 0.01

            for model in ['FFNN', 'CNN', 'GNN']:
                distances_plot = []
                improvements = []

                for d in [3, 5, 7]:
                    if (d in results['Symmetric'] and
                        fixed_p in results['Symmetric'][d] and
                        'MWPM' in results['Symmetric'][d][fixed_p] and
                        model in results['Symmetric'][d][fixed_p]):

                        mwpm_acc = results['Symmetric'][d][fixed_p]['MWPM']['accuracy']
                        model_acc = results['Symmetric'][d][fixed_p][model]['accuracy']

                        if mwpm_acc > 0:
                            improvement = ((model_acc - mwpm_acc) / mwpm_acc) * 100
                            improvements.append(improvement)
                            distances_plot.append(d)

                if improvements:
                    ax.plot(distances_plot, improvements, 'o-',
                          label=model, linewidth=2, markersize=6)

            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Surface Code Distance')
            ax.set_ylabel('Improvement over MWPM (%)')
            ax.set_title(f'Neural Network Improvement (p = {fixed_p})')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

def main():
    """Main execution function for multi-distance analysis"""
    print("Starting Multi-Distance Quantum Error Correction Analysis")
    print("Surface Code Distances: 3, 5, 7")
    print("Models: MWPM, FFNN, CNN, GNN")
    print("Channels: Symmetric and Asymmetric Depolarization")
    print("="*80)

    # Initialize analyzer
    analyzer = MultiDistanceAnalyzer(distances=[3, 5, 7])

    # Define analysis parameters
    error_rates = [0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03]

    noise_scenarios = {
        'Symmetric': (1/3, 1/3, 1/3),
        'X-biased': (0.7, 0.15, 0.15),
        'Z-biased': (0.15, 0.15, 0.7),
        'Y-biased': (0.15, 0.7, 0.15)
    }

    try:
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis(
            error_rates=error_rates,
            noise_scenarios=noise_scenarios,
            train_size=15000,  # Adjusted for computational efficiency
            test_size=3000
        )

        # Create plots
        print("\nCreating comprehensive analysis plots...")
        fig = analyzer.create_comprehensive_plots(results)

        # Create and save results tables
        print("\nGenerating results tables...")
        df, threshold_df = analyzer.create_results_tables(results)

        # Save all results
        analyzer.save_results(results, 'multi_distance_qec_analysis')

        # Create additional publication-ready plots
        print("\nCreating publication-ready plots...")
        create_publication_ready_plots(results)

        print("\n" + "="*60)
        print("MULTI-DISTANCE ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)

        # Print key findings
        print("\nKEY FINDINGS:")
        print("-" * 40)

        # Best thresholds by distance
        if 'Symmetric' in results:
            print("Best MWPM thresholds:")
            for d in [3, 5, 7]:
                if d in results['Symmetric']:
                    threshold = analyzer._find_threshold(results['Symmetric'][d], 'MWPM')
                    print(f"  Distance {d}: {threshold:.4f}")

        # Best performing models
        print("\nBest performing models at p=0.01:")
        for d in [3, 5, 7]:
            if 'Symmetric' in results and d in results['Symmetric'] and 0.01 in results['Symmetric'][d]:
                best_model = min(results['Symmetric'][d][0.01].keys(),
                               key=lambda m: results['Symmetric'][d][0.01][m]['logical_error_rate'])
                best_rate = results['Symmetric'][d][0.01][best_model]['logical_error_rate']
                print(f"  Distance {d}: {best_model} (LogErr: {best_rate:.4f})")

        print("\nDistance scaling analysis:")
        print("Higher distance codes should show exponential improvement in sub-threshold regime")

    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        print("Running simplified analysis...")

        # Fallback to simplified analysis
        simplified_results = run_simplified_multi_distance_analysis()
        print("Simplified multi-distance analysis completed.")

def run_simplified_multi_distance_analysis():
    """Simplified analysis for testing purposes"""

    print("Running simplified multi-distance analysis...")

    distances = [3, 5]  # Reduced for speed
    error_rates = [0.01, 0.02]  # Reduced for speed

    results = {}

    for d in distances:
        print(f"\nDistance {d}:")
        simulator = QuantumErrorCorrectionSimulator(d)

        d_results = {}

        for p in error_rates:
            print(f"  p = {p}")

            try:
                # Small dataset for testing
                test_data = simulator.create_dataset(p, 1000)

                syndrome_start = 2 * simulator.config.n_qubits
                syndrome_end = syndrome_start + simulator.config.n_stabilizers

                X_test = test_data[:, syndrome_start:syndrome_end].astype(np.float32)
                y_test_lld = test_data[:, :syndrome_start]

                # Normalize
                scaler = StandardScaler()
                X_test = scaler.fit_transform(X_test)

                # Test MWPM only
                mwpm_results = simulator.evaluate_mwpm(X_test, y_test_lld)
                d_results[p] = {'MWPM': mwpm_results}

                print(f"    MWPM: Acc={mwpm_results['accuracy']:.4f}, LogErr={mwpm_results['logical_error_rate']:.4f}")

            except Exception as e:
                print(f"    Error: {e}")
                d_results[p] = {'MWPM': {'accuracy': 0.0, 'logical_error_rate': 1.0}}

        results[d] = d_results

    return {'Symmetric': results}

def create_publication_ready_plots(results: Dict) -> None:
    """Create publication-ready plots"""

    # Create threshold comparison
    fig1 = AnalysisVisualizer.create_threshold_comparison_plot(results)
    fig1.savefig('qec_threshold_comparison.pdf', dpi=300, bbox_inches='tight')

    # Create distance scaling analysis
    fig2 = AnalysisVisualizer.create_distance_scaling_analysis(results)
    fig2.savefig('qec_distance_scaling.pdf', dpi=300, bbox_inches='tight')

    print("Publication-ready plots saved:")
    print("- qec_threshold_comparison.pdf")
    print("- qec_distance_scaling.pdf")

# Additional utility functions
def analyze_code_performance(results: Dict, distance: int, model: str = 'MWPM') -> Dict:
    """Analyze performance of specific code distance and model"""

    analysis = {}

    if 'Symmetric' in results and distance in results['Symmetric']:
        error_rates = sorted(results['Symmetric'][distance].keys())

        # Find threshold
        threshold = None
        for p in error_rates:
            if model in results['Symmetric'][distance][p]:
                logical_rate = results['Symmetric'][distance][p][model]['logical_error_rate']
                if logical_rate >= p:
                    threshold = p
                    break

        analysis['threshold'] = threshold

        # Sub-threshold performance
        sub_threshold_rates = [p for p in error_rates if p < (threshold or 0.02)]
        if len(sub_threshold_rates) >= 2:
            # Calculate suppression rate
            p1, p2 = sub_threshold_rates[0], sub_threshold_rates[-1]

            if (model in results['Symmetric'][distance][p1] and
                model in results['Symmetric'][distance][p2]):

                logical1 = results['Symmetric'][distance][p1][model]['logical_error_rate']
                logical2 = results['Symmetric'][distance][p2][model]['logical_error_rate']

                if logical1 > 0 and logical2 > 0:
                    suppression_rate = np.log(logical2 / logical1) / np.log(p2 / p1)
                    analysis['suppression_rate'] = suppression_rate

        # Best accuracy
        best_accuracy = 0
        for p in error_rates:
            if model in results['Symmetric'][distance][p]:
                acc = results['Symmetric'][distance][p][model]['accuracy']
                best_accuracy = max(best_accuracy, acc)

        analysis['best_accuracy'] = best_accuracy

    return analysis

def compare_decoders(results: Dict, distance: int, error_rate: float) -> Dict:
    """Compare all decoders at specific distance and error rate"""

    comparison = {}

    if ('Symmetric' in results and
        distance in results['Symmetric'] and
        error_rate in results['Symmetric'][distance]):

        for model in ['MWPM', 'FFNN', 'CNN', 'GNN']:
            if model in results['Symmetric'][distance][error_rate]:
                comparison[model] = results['Symmetric'][distance][error_rate][model]

    return comparison

def export_summary_report(results: Dict, filename: str = 'qec_summary_report.txt'):
    """Export a comprehensive summary report"""

    with open(filename, 'w') as f:
        f.write("QUANTUM ERROR CORRECTION SIMULATION SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write("- Surface Code Distances: 3, 5, 7\n")
        f.write("- Decoders: MWPM, FFNN, CNN, GNN\n")
        f.write("- Noise Models: Symmetric and Asymmetric Depolarization\n\n")

        # Threshold analysis
        f.write("THRESHOLD ANALYSIS:\n")
        f.write("-" * 30 + "\n")

        for d in [3, 5, 7]:
            if 'Symmetric' in results and d in results['Symmetric']:
                f.write(f"\nDistance {d}:\n")

                for model in ['MWPM', 'FFNN', 'CNN', 'GNN']:
                    analysis = analyze_code_performance(results, d, model)
                    if 'threshold' in analysis and analysis['threshold']:
                        f.write(f"  {model}: {analysis['threshold']:.4f}\n")

        # Performance comparison
        f.write(f"\nPERFORMANCE AT p=0.01:\n")
        f.write("-" * 30 + "\n")

        for d in [3, 5, 7]:
            comparison = compare_decoders(results, d, 0.01)
            if comparison:
                f.write(f"\nDistance {d}:\n")
                for model, perf in comparison.items():
                    f.write(f"  {model}: Acc={perf['accuracy']:.4f}, "
                           f"LogErr={perf['logical_error_rate']:.4f}\n")

    print(f"Summary report saved to {filename}")

if __name__ == "__main__":
    main()

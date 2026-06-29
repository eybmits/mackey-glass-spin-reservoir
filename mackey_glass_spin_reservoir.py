"""
Introductory Mackey-Glass prediction with a quantum spin reservoir.

Run:
    python mackey_glass_spin_reservoir.py

This is a deliberately small Quantum Reservoir Computing (QRC) example.

What makes it a QRC example?

1. The reservoir is a small quantum spin system.
2. Its state is a density matrix, called `rho` in the code.
3. The current scalar input is encoded into one input qubit.
4. The remaining qubits keep their previous quantum state.
5. The spin system evolves with a fixed Hamiltonian.
6. Only the final linear readout is trained.

The most important teaching point:

    The reservoir memory is the density matrix `rho`.

We do not reset `rho` inside the time loop. Each new state depends on the
previous state, so the reservoir carries information forward in time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Easy-to-change teaching settings
# ---------------------------------------------------------------------------

SEED = 3

TRAIN_STEPS = 2500
TEST_STEPS = 1200
SHORT_FREE_RUN_STEPS = 20
WASHOUT_STEPS = 200

N_QUBITS = 5
VIRTUAL_NODES = 7
EVOLUTION_TIME = 0.65
RIDGE = 1e-4


# ---------------------------------------------------------------------------
# Basic quantum objects
# ---------------------------------------------------------------------------

COMPLEX = np.complex128

I2 = np.eye(2, dtype=COMPLEX)
X = np.array([[0, 1], [1, 0]], dtype=COMPLEX)
Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX)


def kron_all(operators):
    """Kronecker product of a list of single-qubit operators."""

    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def one_qubit_operator(n_qubits, qubit, operator):
    """Place a one-qubit operator on one qubit of an n-qubit system."""

    operators = [I2] * n_qubits
    operators[qubit] = operator
    return kron_all(operators)


def two_qubit_operator(n_qubits, qubit_a, operator_a, qubit_b, operator_b):
    """Place two one-qubit operators on two different qubits."""

    operators = [I2] * n_qubits
    operators[qubit_a] = operator_a
    operators[qubit_b] = operator_b
    return kron_all(operators)


def unitary_from_hamiltonian(hamiltonian, time_step):
    """Compute U = exp(-i H dt) using NumPy diagonalization."""

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * eigenvalues * time_step)
    return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T


def expectation(rho, operator):
    """Expectation value Tr(rho O), returned as a real number."""

    return float(np.real(np.trace(rho @ operator)))


# ---------------------------------------------------------------------------
# Mackey-Glass data
# ---------------------------------------------------------------------------


def make_mackey_glass(
    n_points,
    tau=17,
    beta=0.2,
    gamma=0.1,
    power=10,
    dt=1.0,
    seed=7,
):
    """Generate a Mackey-Glass time series with a simple Euler update.

    The continuous equation is:

        dx/dt = beta * x(t - tau) / (1 + x(t - tau)^power) - gamma * x(t)

    The equation uses the delayed value x(t - tau), so the task rewards models
    that can keep useful history in their internal state.
    """

    rng = np.random.default_rng(seed)

    total_length = n_points + tau + 1
    x = np.empty(total_length)

    # A short initial history. The tiny noise avoids a perfectly flat start.
    x[: tau + 1] = 1.2 + 0.01 * rng.normal(size=tau + 1)

    for t in range(tau, total_length - 1):
        delayed_x = x[t - tau]
        dx = beta * delayed_x / (1.0 + delayed_x**power) - gamma * x[t]
        x[t + 1] = x[t] + dt * dx

    return x[tau + 1 :]


# ---------------------------------------------------------------------------
# Quantum spin reservoir
# ---------------------------------------------------------------------------


class QuantumSpinReservoir:
    """A small stateful quantum spin reservoir.

    The reservoir state is a density matrix:

        self.rho

    `self.rho` is the memory. It is updated at every time step and carried into
    the next time step.
    """

    def __init__(self, n_qubits, virtual_nodes, evolution_time, seed):
        self.n_qubits = n_qubits
        self.virtual_nodes = virtual_nodes

        self.hamiltonian = self._make_spin_hamiltonian(seed)
        self.unitary = unitary_from_hamiltonian(self.hamiltonian, evolution_time)
        self.unitary_dagger = self.unitary.conj().T

        dimension = 2**n_qubits

        # Start from a neutral mixed state. After the washout period, the exact
        # initial state is no longer important.
        self.rho = np.eye(dimension, dtype=COMPLEX) / dimension

        # We read out simple Pauli observables from every qubit. These are the
        # measured reservoir features passed to the linear model.
        self.readout_operators = []
        for qubit in range(n_qubits):
            self.readout_operators.append(one_qubit_operator(n_qubits, qubit, Z))
            self.readout_operators.append(one_qubit_operator(n_qubits, qubit, X))

    def _make_spin_hamiltonian(self, seed):
        """Create a fixed disordered quantum spin Hamiltonian.

        This is a small transverse-field spin model with nearest-neighbor and
        weak longer-range couplings. The Hamiltonian is random but fixed by the
        seed, so the example is reproducible.
        """

        rng = np.random.default_rng(seed)
        dimension = 2**self.n_qubits
        hamiltonian = np.zeros((dimension, dimension), dtype=COMPLEX)

        # Local fields.
        for qubit in range(self.n_qubits):
            hamiltonian += rng.uniform(0.3, 1.1) * one_qubit_operator(
                self.n_qubits, qubit, X
            )
            hamiltonian += rng.uniform(-0.25, 0.25) * one_qubit_operator(
                self.n_qubits, qubit, Z
            )

        # Nearest-neighbor spin-spin couplings.
        for qubit in range(self.n_qubits - 1):
            hamiltonian += rng.uniform(0.4, 1.2) * two_qubit_operator(
                self.n_qubits, qubit, Z, qubit + 1, Z
            )
            hamiltonian += rng.uniform(-0.7, 0.7) * two_qubit_operator(
                self.n_qubits, qubit, X, qubit + 1, X
            )

        # Weak longer-range couplings make the dynamics richer.
        for qubit_a in range(self.n_qubits):
            for qubit_b in range(qubit_a + 2, self.n_qubits):
                hamiltonian += rng.uniform(-0.25, 0.25) * two_qubit_operator(
                    self.n_qubits, qubit_a, Z, qubit_b, Z
                )

        return hamiltonian

    def _trace_out_input_qubit(self):
        """Return the reduced density matrix of all qubits except qubit 0."""

        rest_dimension = 2 ** (self.n_qubits - 1)
        rho_reshaped = self.rho.reshape(2, rest_dimension, 2, rest_dimension)

        # Partial trace over the first qubit:
        # sum over <0|rho|0> and <1|rho|1>.
        return rho_reshaped[0, :, 0, :] + rho_reshaped[1, :, 1, :]

    def _input_qubit_state(self, input_value_01):
        """Encode a scalar in [0, 1] as a one-qubit pure state."""

        input_value_01 = float(np.clip(input_value_01, 0.0, 1.0))

        ket = np.array(
            [
                np.sqrt(1.0 - input_value_01),
                np.sqrt(input_value_01),
            ],
            dtype=COMPLEX,
        )

        return np.outer(ket, ket.conj())

    def inject_input(self, input_value_01):
        """Inject the current input into qubit 0.

        This is the standard QRC input step used here:

        1. Replace qubit 0 by a new input state.
        2. Keep the reduced state of the remaining qubits.

        The remaining qubits are not reset. They carry the reservoir memory.
        """

        input_rho = self._input_qubit_state(input_value_01)
        memory_rho = self._trace_out_input_qubit()
        self.rho = np.kron(input_rho, memory_rho)

    def evolve_and_measure(self):
        """Evolve the quantum state and collect virtual-node measurements."""

        features = []

        for _ in range(self.virtual_nodes):
            # This is the stateful quantum evolution:
            # rho(t + dt) = U rho(t) U^\dagger
            self.rho = self.unitary @ self.rho @ self.unitary_dagger

            for operator in self.readout_operators:
                features.append(expectation(self.rho, operator))

        return np.array(features)

    def step(self, input_value_01):
        """Inject one input, evolve, measure, and keep the new quantum state."""

        self.inject_input(input_value_01)
        return self.evolve_and_measure()

    def get_state(self):
        """Save the current density matrix."""

        return self.rho.copy()

    def set_state(self, rho):
        """Restore a saved density matrix."""

        self.rho = rho.copy()


# ---------------------------------------------------------------------------
# Learning utilities
# ---------------------------------------------------------------------------


def readout_features(current_value_z, qrc_measurements):
    """Combine bias, current input, and QRC memory measurements."""

    return np.concatenate(([1.0, current_value_z], qrc_measurements))


def fit_ridge_regression(features, targets, ridge):
    """Fit a linear readout with closed-form ridge regression."""

    identity = np.eye(features.shape[1])
    return np.linalg.solve(
        features.T @ features + ridge * identity,
        features.T @ targets,
    )


def rmse(predictions, targets):
    """Root mean squared error."""

    return np.sqrt(np.mean((predictions - targets) ** 2))


def scale_to_unit_interval(values, train_min, train_max):
    """Map original Mackey-Glass values to [0, 1] for input injection."""

    return np.clip((values - train_min) / (train_max - train_min), 0.0, 1.0)


def denormalize(values_z, mean, std):
    """Map normalized values back to the original Mackey-Glass scale."""

    return values_z * std + mean


def train_qrc(series_z, input_values_01):
    """Run the training sequence once and fit the linear readout."""

    reservoir = QuantumSpinReservoir(
        n_qubits=N_QUBITS,
        virtual_nodes=VIRTUAL_NODES,
        evolution_time=EVOLUTION_TIME,
        seed=SEED,
    )

    collected_features = []
    targets = []

    for t in range(TRAIN_STEPS):
        qrc_measurements = reservoir.step(input_values_01[t])

        # Washout lets the initially neutral quantum state synchronize with the
        # signal before we ask the readout to learn from it.
        if t >= WASHOUT_STEPS:
            collected_features.append(
                readout_features(series_z[t], qrc_measurements)
            )
            targets.append(series_z[t + 1])

    features = np.vstack(collected_features)
    targets = np.array(targets)
    readout_weights = fit_ridge_regression(features, targets, RIDGE)

    return reservoir, readout_weights


def predict_one_step(reservoir, readout_weights, series_z, input_values_01):
    """Teacher-forced prediction: the true current value is always supplied."""

    predictions = []

    for t in range(TRAIN_STEPS, TRAIN_STEPS + TEST_STEPS):
        qrc_measurements = reservoir.step(input_values_01[t])
        features = readout_features(series_z[t], qrc_measurements)
        predictions.append(features @ readout_weights)

    return np.array(predictions)


def predict_short_free_run(
    reservoir,
    readout_weights,
    first_input_z,
    train_z_min,
    train_z_max,
    train_mean,
    train_std,
    train_min,
    train_max,
):
    """Short autonomous rollout.

    Chaotic systems are sensitive to tiny errors. This short rollout is useful
    for teaching the difference between:

    - one-step prediction, where the true current input is supplied
    - free-running prediction, where the model feeds back its own output

    We clip the feedback to the training range before reinjecting it. That keeps
    the demonstration numerically well behaved and easy to inspect.
    """

    predictions = []
    current_value_z = first_input_z

    for _ in range(SHORT_FREE_RUN_STEPS):
        clipped_value_z = float(np.clip(current_value_z, train_z_min, train_z_max))
        clipped_value = denormalize(clipped_value_z, train_mean, train_std)
        input_value_01 = scale_to_unit_interval(clipped_value, train_min, train_max)

        qrc_measurements = reservoir.step(input_value_01)
        features = readout_features(clipped_value_z, qrc_measurements)
        prediction_z = features @ readout_weights
        predictions.append(prediction_z)

        # This line makes the prediction autonomous.
        current_value_z = prediction_z

    return np.array(predictions)


def fit_memoryless_baseline(series_z):
    """Fit y(t+1) from only y(t), with no reservoir state."""

    features = []
    targets = []

    for t in range(WASHOUT_STEPS, TRAIN_STEPS):
        features.append([1.0, series_z[t]])
        targets.append(series_z[t + 1])

    features = np.array(features)
    targets = np.array(targets)
    return fit_ridge_regression(features, targets, RIDGE)


def make_plot(
    one_step_predictions,
    one_step_targets,
    free_predictions,
    free_targets,
    output_path,
):
    """Save a small plot that students can compare with the printed RMSE."""

    n_one_step_to_show = 300

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    axes[0].plot(
        one_step_targets[:n_one_step_to_show],
        label="true Mackey-Glass",
        linewidth=2,
    )
    axes[0].plot(
        one_step_predictions[:n_one_step_to_show],
        label="QRC one-step prediction",
        linestyle="--",
    )
    axes[0].set_title("One-step prediction")
    axes[0].set_ylabel("x(t)")
    axes[0].legend()

    axes[1].plot(free_targets, label="true Mackey-Glass", linewidth=2)
    axes[1].plot(
        free_predictions,
        label="short free-running QRC",
        linestyle="--",
    )
    axes[1].set_title("Short free-running prediction")
    axes[1].set_xlabel("test step")
    axes[1].set_ylabel("x(t)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    total_points_needed = TRAIN_STEPS + TEST_STEPS + 1
    series = make_mackey_glass(total_points_needed)

    # Normalize using only the training part. This avoids leaking test
    # statistics into training.
    train_values = series[:TRAIN_STEPS]
    train_mean = train_values.mean()
    train_std = train_values.std()
    series_z = (series - train_mean) / train_std

    # The quantum input qubit expects a value in [0, 1]. The min and max also
    # come only from the training part.
    train_min = train_values.min()
    train_max = train_values.max()
    input_values_01 = scale_to_unit_interval(series, train_min, train_max)

    reservoir, readout_weights = train_qrc(series_z, input_values_01)
    state_after_training = reservoir.get_state()

    one_step_predictions_z = predict_one_step(
        reservoir,
        readout_weights,
        series_z,
        input_values_01,
    )
    one_step_targets_z = series_z[TRAIN_STEPS + 1 : TRAIN_STEPS + TEST_STEPS + 1]

    # Restore the quantum state from the exact end of training before the short
    # autonomous rollout.
    reservoir.set_state(state_after_training)
    free_predictions_z = predict_short_free_run(
        reservoir,
        readout_weights,
        first_input_z=series_z[TRAIN_STEPS],
        train_z_min=series_z[:TRAIN_STEPS].min(),
        train_z_max=series_z[:TRAIN_STEPS].max(),
        train_mean=train_mean,
        train_std=train_std,
        train_min=train_min,
        train_max=train_max,
    )
    free_targets_z = series_z[
        TRAIN_STEPS + 1 : TRAIN_STEPS + SHORT_FREE_RUN_STEPS + 1
    ]

    baseline_weights = fit_memoryless_baseline(series_z)
    baseline_predictions_z = np.array(
        [
            np.array([1.0, series_z[t]]) @ baseline_weights
            for t in range(TRAIN_STEPS, TRAIN_STEPS + TEST_STEPS)
        ]
    )

    one_step_predictions = denormalize(one_step_predictions_z, train_mean, train_std)
    one_step_targets = denormalize(one_step_targets_z, train_mean, train_std)
    free_predictions = denormalize(free_predictions_z, train_mean, train_std)
    free_targets = denormalize(free_targets_z, train_mean, train_std)

    one_step_rmse_z = rmse(one_step_predictions_z, one_step_targets_z)
    one_step_rmse = rmse(one_step_predictions, one_step_targets)
    free_rmse_z = rmse(free_predictions_z, free_targets_z)
    free_rmse = rmse(free_predictions, free_targets)
    baseline_rmse_z = rmse(baseline_predictions_z, one_step_targets_z)

    repo_dir = Path(__file__).resolve().parent
    results_dir = repo_dir / "results"
    results_dir.mkdir(exist_ok=True)
    plot_path = results_dir / "mackey_glass_qrc_prediction.png"
    metrics_path = results_dir / "metrics.txt"

    make_plot(
        one_step_predictions,
        one_step_targets,
        free_predictions,
        free_targets,
        plot_path,
    )

    summary = f"""Mackey-Glass quantum spin reservoir results

One-step QRC RMSE, normalized scale:       {one_step_rmse_z:.6f}
One-step QRC RMSE, original scale:         {one_step_rmse:.6f}
Short free-run QRC RMSE, normalized scale: {free_rmse_z:.6f}
Short free-run QRC RMSE, original scale:   {free_rmse:.6f}
Memoryless one-step baseline RMSE:         {baseline_rmse_z:.6f}

Reservoir:
- {N_QUBITS} qubits
- {VIRTUAL_NODES} virtual measurement nodes per input
- density-matrix state rho with shape {reservoir.rho.shape}

Plot saved to:
{plot_path}

Notes:
- The reservoir is a quantum spin system.
- The trained model is only the final linear readout.
- The quantum density matrix rho is the model memory.
- rho is never reset inside the time loop.
"""

    print(summary)
    metrics_path.write_text(summary)


if __name__ == "__main__":
    main()


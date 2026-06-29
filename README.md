# Mackey-Glass Quantum Spin Reservoir

This is a deliberately small teaching repository for **Quantum Reservoir
Computing (QRC)**.

It solves a Mackey-Glass next-step prediction task with a small quantum spin
reservoir. The code uses only NumPy and Matplotlib. There is no quantum SDK
dependency, because the goal is to make the mechanics readable.

The main teaching goal is to make **memory** and **statefulness** visible in
code. The important object is:

```python
self.rho
```

`self.rho` is the quantum density matrix of the reservoir. It is carried from
one time step to the next. It is not reset inside the prediction loop. Because
every new quantum state depends on the previous quantum state, the reservoir
contains memory of the recent input history.

## Files

- `mackey_glass_spin_reservoir.py` - the full example, written to be read top to bottom
- `requirements.txt` - minimal Python dependencies
- `results/` - created automatically when the script is run

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python mackey_glass_spin_reservoir.py
```

The script prints RMSE values and saves a plot to:

```text
results/mackey_glass_qrc_prediction.png
```

## What Makes This QRC?

The reservoir is a small quantum spin system with a fixed Hamiltonian:

```python
rho(t + dt) = U rho(t) U^\dagger
```

At each time step:

1. The current Mackey-Glass value is encoded into qubit 0.
2. The other qubits keep their reduced quantum state.
3. The whole spin system evolves under a fixed Hamiltonian.
4. Pauli `X` and `Z` expectation values are measured.
5. A linear readout predicts the next Mackey-Glass value.

Only the linear readout is trained. The quantum reservoir itself is fixed after
random initialization.

## Where Memory Lives

In the code, the key state update is:

```python
self.rho = self.unitary @ self.rho @ self.unitary_dagger
```

This line is stateful because the new `self.rho` depends on the old `self.rho`.
That old quantum state contains information from previous inputs.

The input injection is:

```python
self.rho = np.kron(input_rho, memory_rho)
```

Here `input_rho` is the new input qubit. `memory_rho` is the reduced density
matrix of the other qubits. Those other qubits are the memory part of the
reservoir.

## Expected Result

With the default seed and settings, the example should produce approximately:

```text
One-step QRC RMSE, normalized scale:       0.0053
One-step QRC RMSE, original scale:         0.0012
Short free-run QRC RMSE, normalized scale: 0.077
Short free-run QRC RMSE, original scale:   0.018
Memoryless one-step baseline RMSE:         0.147
```

Small numerical differences are normal across machines.

The one-step score is the main introductory benchmark. The short free-running
score is included to show what changes when the model feeds its own prediction
back as the next input. For chaotic systems, long free-running forecasts
eventually drift even when one-step prediction is very accurate.

## Teaching Notes

- `N_QUBITS` controls the quantum reservoir size.
- `VIRTUAL_NODES` controls how many intermediate measurements are collected per input.
- `EVOLUTION_TIME` controls how long the quantum system evolves between virtual nodes.
- `RIDGE` regularizes the final linear readout.
- The memoryless baseline is much worse because it only sees the current value,
  not the quantum reservoir state.


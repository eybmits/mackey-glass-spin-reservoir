# Student Guide

This repository shows one complete Quantum Reservoir Computing example:

```text
Mackey-Glass signal -> quantum spin reservoir -> linear readout -> next value
```

The goal is not to build the largest or fastest QRC model. The goal is to make
the memory mechanism easy to see in code.

## The Big Idea

Mackey-Glass is a delayed chaotic time series. The next value depends on past
values, so a useful model needs memory.

In this code, the memory is the quantum reservoir state:

```python
self.rho
```

`rho` is a density matrix. It describes the current quantum state of the spin
system.

## One Time Step

At every time step, the code does this:

```text
1. Put the current input value into qubit 0.
2. Keep the old quantum state of the other qubits.
3. Evolve the whole quantum spin system.
4. Measure Pauli X and Z expectation values.
5. Use a linear readout to predict the next Mackey-Glass value.
```

The key memory point is step 2. The other qubits are not reset.

## Why This Is Stateful

A stateless model forgets everything after each input.

This QRC model does not forget immediately, because the next quantum state uses
the previous quantum state:

```python
self.rho = self.unitary @ self.rho @ self.unitary_dagger
```

That line means:

```text
new quantum state = quantum evolution of old quantum state
```

So the reservoir carries information forward through time.

## What Gets Trained?

Only the final linear readout is trained.

The quantum spin reservoir is fixed after initialization. This is the usual
reservoir-computing idea:

```text
fixed rich dynamics + simple trained readout
```

## What Result Should You See?

After running:

```bash
python mackey_glass_spin_reservoir.py
```

you should see approximately:

```text
One-step QRC RMSE, normalized scale:       0.005254
Memoryless one-step baseline RMSE:         0.146814
```

The QRC error is much smaller than the memoryless baseline because the QRC has a
state that remembers recent history.

## What To Try Changing

Start with these constants near the top of the Python file:

```python
N_QUBITS = 5
VIRTUAL_NODES = 7
EVOLUTION_TIME = 0.65
RIDGE = 1e-4
```

Good exercises:

1. Set `N_QUBITS = 4`. Does the RMSE get worse?
2. Set `VIRTUAL_NODES = 3`. What changes?
3. Increase `RIDGE`. Does the prediction become smoother?
4. Comment out the reservoir features and use only the current input. This
   should behave like the memoryless baseline.


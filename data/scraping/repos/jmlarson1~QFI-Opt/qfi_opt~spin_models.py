#!/usr/bin/env python3
import argparse
import functools
import itertools
import os
import sys
from typing import Callable, Optional, Sequence

import qfi_opt
from qfi_opt.dissipation import Dissipator

DISABLE_JAX = bool(os.getenv("DISABLE_JAX"))

if not DISABLE_JAX:
    import jax
    import jax.numpy as np

    jax.config.update("jax_enable_x64", True)

else:
    import numpy as np  # type: ignore[no-redef]
    import scipy

COMPLEX_TYPE = np.complex128
DEFAULT_DISSIPATION_FORMAT = "XYZ"

# qubit/spin states
KET_0 = np.array([1, 0], dtype=COMPLEX_TYPE)  # |0>, spin up
KET_1 = np.array([0, 1], dtype=COMPLEX_TYPE)  # |1>, spin down

# Pauli operators
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX_TYPE)  # |0><0| - |1><1|
PAULI_X = np.array([[0, 1], [1, 0]], dtype=COMPLEX_TYPE)  # |0><1| + |1><0|
PAULI_Y = -1j * PAULI_Z @ PAULI_X


def log2_int(val: int) -> int:
    return val.bit_length() - 1


def simulate_sensing_protocol(
    params: Sequence[float] | np.ndarray,
    entangling_hamiltonian: np.ndarray,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """
    Simulate a sensing protocol, and return the final state (density matrix).

    Starting with an initial all-|1> state (all spins pointing down along the Z axis):
    1. Rotate about an axis in the XY plane.
    2. Evolve under a given entangling Hamiltonian.
    3. "Un-rotate" about an axis in the XY plane.

    Step 1 rotates by an angle '+np.pi * params[1]', about the axis 'np.pi * params[2]'.
    Step 2 evolves under the given entangling Hamiltonian for time 'params[0] * num_qubits * np.pi'.
    Step 3 rotates by an angle '-np.pi * params[3]', about the axis 'np.pi * params[4]'.

    If dissipation_rates is nonzero, qubits experience dissipation during the entangling step (2).
    See the documentation for the Dissipator class for a general explanation of the
    dissipation_rates and dissipation_format arguments.

    This method additionally divides the dissipator (equivalently, all dissipation rates) by a
    factor of 'np.pi * num_qubits' in order to "normalize" dissipation time scales, and make them
    comparable to the time scales of coherent evolution.  Dividing a Dissipator with homogeneous
    dissipation rates 'r' by a factor of 'np.pi * num_qubits' makes so that each qubit depolarizes
    with probability 'e^(-params[0] * r)' by the end of the OAT protocol.
    """
    assert len(params) == 5, "Spin sensing protocols accept five parameters."

    num_qubits = log2_int(entangling_hamiltonian.shape[0])

    # construct collective spin operators
    collective_Sx, collective_Sy, collective_Sz = collective_spin_ops(num_qubits)

    # rotate the all-|1> state about a chosen axis
    time_1 = params[1] * np.pi
    axis_angle_1 = params[2] * np.pi
    qubit_ket = np.sin(time_1 / 2) * KET_0 + 1j * np.exp(1j * axis_angle_1) * np.cos(time_1 / 2) * KET_1
    qubit_state = np.outer(qubit_ket, qubit_ket.conj())
    state_1 = functools.reduce(np.kron, [qubit_state] * num_qubits)

    # entangle!
    time_2 = params[0] * np.pi * num_qubits
    dissipator = Dissipator(dissipation_rates, dissipation_format) / (np.pi * num_qubits)
    state_2 = evolve_state(state_1, time_2, entangling_hamiltonian, dissipator)

    # un-rotate about a chosen axis
    time_3 = -params[3] * np.pi
    axis_angle_3 = params[4] * np.pi
    final_hamiltonian = np.cos(axis_angle_3) * collective_Sx + np.sin(axis_angle_3) * collective_Sy
    state_3 = evolve_state(state_2, time_3, final_hamiltonian)

    return state_3


def enable_axial_symmetry(simulate_func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Decorator to enable an axially-symmetric version of a simulation method.

    Axial symmetry means that the second parameter (which controls the axis of the first rotation) can be set to zero without loss of generality.
    For this reason, if the simulation method is run with `axial_symmetry=True` (which this decorator sets by default), then the method accepts only
    four parameters rather than the usual five.  The method additionally checks that the dissipation rates respect the axial symmetry, if applicable.
    """

    def simulate_func_with_symmetry(params: Sequence[float] | np.ndarray, *args: object, axial_symmetry: bool = True, **kwargs: object) -> np.ndarray:
        if axial_symmetry:
            # Verify that dissipation satisfies axial symmetry.
            dissipation_rates = kwargs.get("dissipation_rates", 0.0)
            dissipation_format = kwargs.get("dissipation_format", DEFAULT_DISSIPATION_FORMAT)
            if dissipation_format == "XYZ" and hasattr(dissipation_rates, "__iter__"):
                rate_sx, rate_sy, *_ = dissipation_rates
                if not rate_sx == rate_sy:
                    raise ValueError(
                        f"Dissipation format {dissipation_format} with rates {dissipation_rates} does not respect axial symmetry."
                        "\nTry passing the argument `axial_symmetry=False` to the simulation method."
                    )

            # If there are only four parameters, (entangling_time, initial_rotation_angle, initial_rotation_axis, final_rotation_angle),
            # append a final_rotation_axis of 0.
            if len(params) == 4:
                params = np.append(np.array(params), 0.0)

        return simulate_func(params, *args, **kwargs)

    return simulate_func_with_symmetry


@enable_axial_symmetry
def simulate_OAT(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate a one-axis twisting (OAT) protocol."""
    _, _, collective_Sz = collective_spin_ops(num_qubits)
    hamiltonian = collective_Sz.diagonal() ** 2 / num_qubits
    return simulate_sensing_protocol(
        params,
        hamiltonian,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


def simulate_TAT(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate a two-axis twisting (TAT) protocol."""
    collective_Sx, collective_Sy, _ = collective_spin_ops(num_qubits)
    hamiltonian = (collective_Sx @ collective_Sy + collective_Sy @ collective_Sx) / num_qubits
    return simulate_sensing_protocol(
        params,
        hamiltonian,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


def simulate_spin_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_op: np.ndarray,
    coupling_exponent: float = 0.0,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate an entangling protocol for a spin chain with power-law interactions."""
    normalization_factor = num_qubits * np.array([1 / abs(pp - qq) ** coupling_exponent for pp, qq in itertools.combinations(range(num_qubits), 2)]).mean()
    hamiltonian = sum(
        act_on_subsystem(num_qubits, coupling_op, pp, qq) / abs(pp - qq) ** coupling_exponent for pp, qq in itertools.combinations(range(num_qubits), 2)
    )
    return simulate_sensing_protocol(
        params,
        hamiltonian / normalization_factor,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


@enable_axial_symmetry
def simulate_ising_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_exponent: float = 0.0,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = np.kron(PAULI_Z, PAULI_Z) / 2
    return simulate_spin_chain(
        params,
        num_qubits,
        coupling_op,
        coupling_exponent,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


@enable_axial_symmetry
def simulate_XX_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_exponent: float = 0.0,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = (np.kron(PAULI_X, PAULI_X) + np.kron(PAULI_Y, PAULI_Y)) / 2
    return simulate_spin_chain(
        params,
        num_qubits,
        coupling_op,
        coupling_exponent,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


def simulate_local_TAT_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_exponent: float = 0.0,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = (np.kron(PAULI_X, PAULI_Y) + np.kron(PAULI_Y, PAULI_X)) / 2
    return simulate_spin_chain(
        params,
        num_qubits,
        coupling_op,
        coupling_exponent,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


def evolve_state(
    density_op: np.ndarray,
    time: float | np.ndarray,
    hamiltonian: np.ndarray,
    dissipator: Optional[Dissipator] = None,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    disable_jax: bool = DISABLE_JAX,
) -> np.ndarray:
    """
    Time-evolve a given initial density operator for a given amount of time under the given Hamiltonian and (optionally) Dissipator.
    """
    if time.real == 0:
        return density_op

    # treat negative times as evolving under the negative of the Hamiltonian
    # NOTE: this is required for autodiff to work
    if time.real < 0:
        time, hamiltonian = -time, -hamiltonian

    times = np.linspace(0.0, time, 2)
    time_deriv = get_time_deriv(hamiltonian, dissipator)

    if not DISABLE_JAX:
        result = qfi_opt.ode_jax.odeint(
            time_deriv,
            density_op,
            times,
            rtol=rtol,
            atol=atol,
        )
        return result[-1]

    def scipy_time_deriv(time: float, density_op: np.ndarray) -> np.ndarray:
        density_op.shape = (hamiltonian.shape[0],) * 2  # type: ignore[misc]
        output = time_deriv(density_op, time)
        density_op.shape = (-1,)  # type: ignore[misc]
        return output.ravel()

    result = scipy.integrate.solve_ivp(
        scipy_time_deriv,
        times.real,
        density_op.ravel(),
        rtol=rtol,
        atol=atol,
    )
    return result.y[:, -1].reshape(density_op.shape)


def get_time_deriv(
    hamiltonian: np.ndarray,
    dissipator: Optional[Dissipator] = None,
    *,
    disable_jax: bool = DISABLE_JAX,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Construct a time derivative function that maps (state, time) --> d(state)/d(time)."""

    # construct the time derivative from coherent evolution
    if hamiltonian.ndim == 2:
        # ... with ordinary matrix multiplication
        def coherent_time_deriv(density_op: np.ndarray, time: float) -> np.ndarray:
            return -1j * (hamiltonian @ density_op - density_op @ hamiltonian)

    else:
        # 'hamiltonian' is a 1-D array of the values on the diagonal of the actual Hamiltonian,
        # so we can compute the commutator with array broadcasting, which is faster than matrix multiplication
        expanded_hamiltonian = np.expand_dims(hamiltonian, 1)

        def coherent_time_deriv(density_op: np.ndarray, time: float) -> np.ndarray:
            return -1j * (expanded_hamiltonian * density_op - density_op * hamiltonian)

    if not dissipator:
        return coherent_time_deriv

    def dissipative_time_deriv(density_op: np.ndarray, time: float) -> np.ndarray:
        return coherent_time_deriv(density_op, time) + dissipator @ density_op

    return dissipative_time_deriv


@functools.cache
def collective_spin_ops(num_qubits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct collective spin operators."""
    return (
        collective_op(PAULI_X, num_qubits) / 2,
        collective_op(PAULI_Y, num_qubits) / 2,
        collective_op(PAULI_Z, num_qubits) / 2,
    )


def collective_op(op: np.ndarray, num_qubits: int) -> np.ndarray:
    """Compute the collective version of a single-qubit qubit operator: sum_q op_q."""
    assert op.shape == (2, 2)
    return sum((act_on_subsystem(num_qubits, op, qubit) for qubit in range(num_qubits)), start=np.array(0))


def act_on_subsystem(num_qubits: int, op: np.ndarray, *qubits: int) -> np.ndarray:
    """
    Return an operator that acts with 'op' in the given qubits, and trivially (with the identity operator) on all other qubits.
    """
    assert op.shape == (2 ** len(qubits),) * 2, "Operator shape {op.shape} is inconsistent with the number of target qubits provided, {num_qubits}!"
    identity = np.eye(2 ** (num_qubits - len(qubits)), dtype=op.dtype)
    system_op = np.kron(op, identity)

    # rearrange operator into tensor factors addressing each qubit
    system_op = np.moveaxis(
        system_op.reshape((2,) * 2 * num_qubits),
        range(num_qubits),
        range(0, 2 * num_qubits, 2),
    ).reshape((4,) * num_qubits)

    # move the first len(qubits) tensor factors to the target qubits
    system_op = np.moveaxis(
        system_op,
        range(len(qubits)),
        qubits,
    )

    # split and re-combine tensor factors again to recover the operator as a matrix
    return np.moveaxis(
        system_op.reshape((2,) * 2 * num_qubits),
        range(0, 2 * num_qubits, 2),
        range(num_qubits),
    ).reshape((2**num_qubits,) * 2)


def get_jacobian_func(
    simulate_func: Callable,
    *,
    disable_jax: bool = DISABLE_JAX,
    step_sizes: float | Sequence[float] = 1e-4,
) -> Callable:
    """Convert a simulation method into a function that returns its Jacobian."""

    if not DISABLE_JAX:
        jacobian_func = jax.jacrev(simulate_func, argnums=(0,), holomorphic=True)

        def get_jacobian(*args: object, **kwargs: object) -> np.ndarray:
            return jacobian_func(*args, **kwargs)[0]

        return get_jacobian

    def get_jacobian_manually(params: Sequence[float] | np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        nonlocal step_sizes
        if isinstance(step_sizes, float):
            step_sizes = [step_sizes] * len(params)
        assert len(step_sizes) == len(params)

        result_at_params = simulate_func(params, *args, **kwargs)
        shifted_results = []
        for idx, step_size in enumerate(step_sizes):
            new_params = list(params)
            new_params[idx] += step_size
            result_at_params_with_step = simulate_func(new_params, *args, **kwargs)
            shifted_results.append((result_at_params_with_step - result_at_params) / step_size)

        return np.stack(shifted_results, axis=-1)

    return get_jacobian_manually


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Simulate a simple one-axis twisting (OAT) protocol.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--dissipation", type=float, default=0.0)
    parser.add_argument("--params", type=float, nargs=4, default=[0.5, 0.5, 0.5, 0])
    parser.add_argument("--jacobian", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])

    # convert the parameters into a complex array, which is necessary for autodiff capabilities
    args.params = np.array(args.params, dtype=COMPLEX_TYPE)

    if args.jacobian:
        get_jacobian = get_jacobian_func(simulate_OAT)
        jacobian = get_jacobian(args.params, args.num_qubits, dissipation_rates=args.dissipation)
        for pp in range(len(args.params)):
            print(f"d(final_state/d(params[{pp}]):")
            print(jacobian[:, :, pp])

    # simulate the OAT protocol
    final_state = simulate_OAT(args.params, args.num_qubits, dissipation_rates=args.dissipation)

    # compute collective Pauli operators
    mean_X = collective_op(PAULI_X, args.num_qubits) / args.num_qubits
    mean_Y = collective_op(PAULI_Y, args.num_qubits) / args.num_qubits
    mean_Z = collective_op(PAULI_Z, args.num_qubits) / args.num_qubits
    mean_ops = [mean_X, mean_Y, mean_Z]

    # print out expectation values and variances
    final_pauli_vals = np.array([(final_state @ op).trace().real for op in mean_ops])
    final_pauli_vars = np.array([(final_state @ (op @ op)).trace().real - mean_op_val**2 for op, mean_op_val in zip(mean_ops, final_pauli_vals)])
    print("[<X>, <Y>, <Z>]:", final_pauli_vals)
    print("[var(X), var(Y), var(Z)]:", final_pauli_vars)

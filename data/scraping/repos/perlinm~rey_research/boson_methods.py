#!/usr/bin/env python3

import numpy as np
import scipy.integrate

from dicke_methods import coherent_spin_state, spin_op_z_dicke

##########################################################################################
# basic objects and methods for boson mean-field theory simulations
##########################################################################################

# construct a boson MFT state from a quantum state
def boson_mft_state(bare_quantum_state, spin_dim, spin_num):
    quantum_state = np.array(bare_quantum_state).T.copy().astype(complex)

    # if we were given a state of a single spin,
    #   construct uniform product state of all spins
    if quantum_state.ndim == 1:
        quantum_state /= np.sqrt(sum(abs(quantum_state)**2))
        quantum_state = np.repeat(quantum_state, spin_num)
        quantum_state.shape = (spin_dim,spin_num)
        return quantum_state

    # if we were given many states, construct the state of each spin independently
    if quantum_state.ndim == 2:
        quantum_state /= np.sqrt(np.sum(abs(quantum_state)**2, axis = 0))
        return quantum_state

# return a quantum "coherent" (polarized) state of a single spin
def spin_state(direction, spin_dim):
    return coherent_spin_state(direction, spin_dim-1)

# return a polarized state of all spins
def polarized_state(direction, spin_dim, spin_num):
    return boson_mft_state(spin_state(direction, spin_dim), spin_dim, spin_num)

# method to construct a field tensor
def field_tensor(field_ops):
    return np.array(field_ops).transpose([1,2,0])

# method computing the time derivative of a mean-field bosonic state
def boson_time_deriv(state, field, coupling_op = -1):
    spin_dim, spin_num = state.shape
    state_triplet = ( state.conj(), state, state )

    if np.isscalar(coupling_op) or coupling_op.ndim == 0:
        # uniform SU(n)-symmetric couplings
        coupling_vec = coupling_op * np.einsum("nk,ak,ni->ai", *state_triplet)
    elif coupling_op.ndim == 2:
        # inhomogeneous SU(n)-symmetric couplings
        coupling_vec = np.einsum("ik,rk,sk,ni->ai", coupling_op, *state_triplet)
    elif coupling_op.ndim == 4:
        # uniform asymmetric couplings
        coupling_vec = np.einsum("anrs,rk,sk,ni->ai", coupling_op, *state_triplet)
    elif coupling_op.ndim == 6:
        # inhomogeneous asymmetric couplings
        coupling_vec = np.einsum("anirsk,rk,sk,ni->ai", coupling_op, *state_triplet)
    else:
        # couplings that factorize into an "operator" part and a "spatial" part
        coupling_vec = np.einsum("anrs,ik,rk,sk,ni->ai", *coupling_op, *state_triplet)

    if type(field) in [ list, tuple ]:
        # homogeneous operator with inhomogeneous coefficients
        field_op, field_coef = field
        if field_op.ndim == 1:
            # diagonal operator
            field_vec = np.einsum("a,i,ai->ai", field_op, field_coef, state)
        elif field_op.ndim == 2:
            # general operator
            field_vec = np.einsum("an,i,ni->ai", field_op, field_coef, state)
    elif field.ndim == 1:
        # homogeneous field with diagonal operator
        field_vec = np.einsum("n,ni->ai", field, state)
    elif field.shape == state.shape:
        # inhomogeneous field with diagonal operator
        field_vec = np.einsum("ai,ai->ai", field, state)
    elif field.ndim == 3:
        # general inhomogeneous field
        field_vec = np.einsum("ani,ni->ai", field, state)

    return -1j * ( coupling_vec/spin_num + field_vec )

# wrapper for the numerical integrator, for dealing with multi-spin_dimensional state
def evolve_mft(initial_state, times, field, coupling_op = -1, ivp_tolerance = 1e-10):
    state_shape = initial_state.shape
    initial_state.shape = initial_state.size

    def time_deriv_flat(time, state):
        state.shape = state_shape
        vec = boson_time_deriv(state, field, coupling_op).ravel()
        state.shape = state.size
        return vec

    ivp_args = [ time_deriv_flat, (times[0], times[-1]), initial_state ]
    ivp_kwargs = dict( t_eval = times, rtol = ivp_tolerance, atol = ivp_tolerance )
    ivp_solution = scipy.integrate.solve_ivp(*ivp_args, **ivp_kwargs)

    times = ivp_solution.t
    states = ivp_solution.y
    states.shape = state_shape + (times.size,)
    states = states.transpose([2,0,1])

    initial_state.shape = state_shape
    return states

##########################################################################################
# saving and loading simulated states
##########################################################################################

import itertools

# "upper triangular" indices for a density matrix \rho and its 2nd moment \rho\otimes\rho
def tri_indices(spin_dim):
    yield from itertools.combinations_with_replacement(range(spin_dim), r = 2)
def var_indices(spin_dim):
    for mm in range(spin_dim):
        for nn, aa, bb in itertools.product(range(mm, spin_dim), repeat = 3):
            yield mm, nn, aa, bb

# convert a mean-field (pure product) state into
#  (i) a space-averaged density matrix \bar\rho
# (ii) the second moment \bar\rho \otimes \bar\rho
def compute_avg_var_vals(state_MF):
    if state_MF.ndim == 2:
        spin_dim, spin_num = state_MF.shape
        # avg_state = np.einsum("mq,nq->mn", state_MF, state_MF.conj()) / spin_num
        avg_state = ( state_MF @ state_MF.conj().T ) / spin_num
        vals = np.array([ avg_state[mm,nn] for mm, nn in tri_indices(spin_dim) ] +
                        [ avg_state[mm,nn] * avg_state[aa,bb]
                          for mm, nn, aa, bb in var_indices(spin_dim) ])
    elif state_MF.ndim == 3:
        vals = np.mean([ compute_avg_var_vals(_state_MF)
                         for _state_MF in state_MF ],
                       axis = 0)
    if np.allclose(vals, vals.real): vals = vals.real
    return vals

# convert a vector of the "upper-triangular entries" of a density matrix \rho
#   and its second moment \rho\otimes\rho into the full matrices
def extract_avg_var_state(avg_var_vals, spin_dim):
    num_tri_vals = len(list(tri_indices(spin_dim)))
    avg_vals = avg_var_vals[:num_tri_vals]
    var_vals = avg_var_vals[num_tri_vals:]
    avg_state = np.empty((spin_dim,)*2, dtype = avg_var_vals.dtype)
    var_state = np.empty((spin_dim,)*4, dtype = avg_var_vals.dtype)
    for val, ( mm, nn ) in zip(avg_vals, tri_indices(spin_dim)):
        avg_state[mm,nn] = val
        avg_state[nn,mm] = val.conj()
    for val, ( mm, nn, aa, bb ) in zip(var_vals, var_indices(spin_dim)):
        var_state[mm,nn,aa,bb] = var_state[aa,bb,mm,nn] = val
        var_state[nn,mm,bb,aa] = var_state[bb,aa,nn,mm] = val.conj()
    return avg_state, var_state

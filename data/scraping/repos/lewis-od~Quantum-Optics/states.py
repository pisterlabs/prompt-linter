r"""
Methods for generating different types of state.

In all of the below text, :math:`\mathcal{N}` is a normalisation constant,
:math:`\widehat{S}(z)` is the single-mode squeezing operator, and
:math:`\lvert \alpha \rangle` are :func:`~qutip.states.coherent` states.
"""
from qutip import Qobj
from qutip.states import coherent, basis
from qutip.operators import squeeze, position
import numpy as np

# The types of state provided by this module
TYPES = ['cat', 'zombie', 'squeezed_cat', 'cubic_phase', 'on', 'useless']

def cat(T, alpha, theta=0):
    r"""
    Generates a  normalised cat state of the form:

    .. math::

        \lvert \text{cat} \rangle_{\theta} = \mathcal{N} \left ( \lvert \alpha \
        \rangle + e^{i\theta} \lvert -\alpha \rangle \right )

    :param T: The truncation to use
    :param alpha: The complex number prametrising the coherent states
    :param theta: The phase differece between the coherent states
    :returns: A :class:`qutip.Qobj` instance
    """
    a = coherent(T, alpha)
    b = np.exp(1j*theta) * coherent(T, -alpha)
    return (a + b).unit()

def zombie(T, alpha):
    r"""
    Generates a normalised zombie cat state of the form:

    .. math::

        \lvert \text{zombie} \rangle = \mathcal{N} \left( \lvert \alpha \rangle + \
        \lvert e^{2 \pi i /3} \alpha \rangle + \
        \lvert e^{4 \pi i /3} \alpha \rangle \right )

    :param T: The truncation to use
    :param alpha: Complex number parametrising the coherent state
    :returns: A :class:`qutip.Qobj` instance
    """
    a = coherent(T, alpha)
    b = coherent(T, np.exp(2j*np.pi/3)*alpha)
    c = coherent(T, np.exp(4j*np.pi/3)*alpha)
    return (a + b + c).unit()

def squeezed(T, z):
    r"""
    Generates a squeezed state

    .. math::
        \lvert z \rangle = \widehat{S}(z) \lvert 0 \rangle

    :param T: The truncation to use
    :param z: The squeezing parameter
    :returns: A :class:`qutip.Qobj` instance
    """
    vac = basis(T, 0)
    S = squeeze(T, z)
    return S * vac

def squeezed_cat(T, alpha, z):
    r"""
    Generates a squeezed cat state. This is done by generating a perfect
    :func:`~quoptics.states.cat` state (i.e. :math:`\theta=0`), then applying
    the single-mode squeezing operator.

    .. math::
        \lvert \text{cat}, z \rangle = \widehat{S}(z) \lvert \text{cat} \rangle

    :param T: The truncation to use
    :param alpha: The complex number parametrising the cat state
    :param z: The sqeezing parameter
    :returns: A :class:`qutip.Qobj` instance
    """
    c = cat(T, alpha)
    S = squeeze(T, z)
    return (S * c).unit()

def cubic_phase(T, gamma, r):
    r"""
    Generates a finitely squeezed approximation to a cubic phase state:

    .. math::

        \lvert \gamma , z \rangle = e^{i \gamma \widehat{q}^3} \widehat{S}(-r)\
        \lvert 0 \rangle

    Note that exact (unphysical) cubic phase states are given by:

    .. math::

        \lvert \gamma \rangle = e^{i \gamma \widehat{q}^3} \lvert 0 \rangle_{p}
        \\
        \lvert 0 \rangle_{p} = \lim_{r \to \infty} \widehat{S}(-r) \lvert 0 \rangle

    :param T: The truncation to use
    :param gamma: The parameter of the cubic phase operator
    :param r: The parameter of the squeezing operator (positive real number)
    :returns: A :class:`qutip.Qobj` instance
    """
    q = position(T)
    V = (1j*gamma*q**3).expm()
    S = squeeze(T, -r)
    vac = basis(T, 0)
    return (V * S * vac)

def on_state(T, n, delta):
    r"""
    Generates a normalised ON state of the form:

    .. math::

        \lvert \text{ON} \rangle = \mathcal{N} \left ( \lvert 0 \rangle + \
        \delta \lvert N \rangle \right )

    :param T: The truncation to use
    :param N: The Fock state to take a superposition of the vacuum with
    :param delta: Coefficient of the non-vacuum Fock state
    :returns: A :class:`qutip.Qobj` instance
    """
    O = basis(T, 0)
    N = basis(T, n)
    return (O + delta*N).unit()

def useless(T):
    r"""
    Generates a random, normalised 'useless' state. Values towards the end of
    the state vector are exponentially less likely to be populated, such that
    the first entry is definitely populated, and the last entry has a 50% chance
    of being populated.

    :param T: The truncation to use
    :returns: A :class:`qutip.Qobj` instance
    """
    data = np.zeros(T)
    rands = np.random.rand(T)
    k = (T-1)/np.log(0.5)
    rands *= np.array([np.exp(n/k) for n in range(0, T)])
    flips = np.round(rands)
    n_values = int(np.sum(flips))
    data[flips == 1] = np.random.rand(n_values)
    return Qobj(data).unit()

class StateIterator(object):
    r"""
    An `iterator <https://docs.python.org/3/glossary.html#term-iterator>`_
    object that generates random states.

    :param batch_size: How many states to generate
    :param T: The truncation to when generating each state
    :param cutoff: The length of the generated state arrays
        (requires qutip=False)
    :param qutip: If True, states are generated as :class:`qutip.Qobj`,
        otherwise they're generated as numpy arrays
    :ivar types: An array of strings containing the names of types of state to
        generate
    """
    def __init__(self, batch_size, T=100, cutoff=25, qutip=True):
        self.n = 0
        self.batch_size = batch_size
        self.T = T
        self.cutoff = cutoff
        self.qutip = qutip
        self.types = TYPES

    def __iter__(self):
        return self

    def _rand_complex(self, modulus):
        r = np.random.rand() * modulus
        theta = np.random.rand() * np.pi * 2
        z = r * np.exp(1j*theta)
        return z

    def __next__(self):
        label = self.n % len(self.types)
        type = self.types[label]

        if type == 'cat':
            # Choose sign of cat state at random
            theta = np.random.rand() * np.pi * 2
            alpha = self._rand_complex(2.0)
            state = cat(self.T, alpha, theta)
        elif type == 'zombie':
            alpha = self._rand_complex(2.0)
            state = zombie(self.T, alpha)
        elif type == 'squeezed_cat':
            alpha = self._rand_complex(2.0)
            z = self._rand_complex(1.4)
            state = squeezed_cat(self.T, alpha, z)
        elif type == 'cubic_phase':
            gamma = np.random.rand() * 0.25
            r = np.random.rand() * 1.4
            state = cubic_phase(self.T, gamma, r)
        elif type == 'on':
            n = np.random.randint(1, self.cutoff)
            delta = np.random.rand()
            state = on_state(self.T, n, delta)
        elif type == 'useless':
            state = useless(self.T)
        else:
            raise ValueError("Invalid type supplied")

        if self.n == self.batch_size:
            self.n = 0
            raise StopIteration

        if not self.qutip:
            state = to_numpy(state)

        self.n += 1
        return state, label

def random_states(T, n, cutoff=25, qutip=True):
    r"""
    Returns n randomly generated states and their labels

    :param T: The truncation to use when generating the states
    :param n: How many states to generate
    :param cutoff: The length of the generated state vectors
    :param qutip: Whether to return states as :class:`qutip.Qobj` or numpy
        arrays
    :returns: A tuple containing an array of states and an array of labels
    """
    data = [x for x in StateIterator(n, T=T, cutoff=cutoff, qutip=qutip)]
    states, labels = zip(*data)
    if qutip:
        states = np.array(states)
    else:
        states = np.array([s[:cutoff] for s in states])
    labels = np.array(labels)
    return states, labels

def to_numpy(state):
    r"""
    Convert a :class:`qutip.Qobj` instance to a numpy array, formatted so that
    :class:`~quoptics.network.NeuralNetwork` can interpret it.

    The nth entry of the array is the modulus of the coefficient of the nth Fock
    state.

    :param state: A :class:`qutip.Qobj` instace
    :returns: A numpy array containing the state data
    """
    return np.abs(state.data.toarray().T[0])

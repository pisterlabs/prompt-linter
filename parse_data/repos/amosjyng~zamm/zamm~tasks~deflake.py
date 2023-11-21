from langchain.prompts import PromptTemplate

ADD_TYPES_AND_DOCUMENT_TEMPLATE = PromptTemplate.from_template(
    '''
Original code:

```python
def test_addition():
    assert 1 + 1 == 2


def test_subtraction():
    assert 2 - 1 == 1
```

Typed and documented code:

```python
"""Test arithmetic operations."""


def test_addition() -> None:
    """Test addition of two numbers."""
    assert 1 + 1 == 2


def test_subtraction() -> None:
    """Test subtraction of two numbers."""
    assert 2 - 1 == 1
```

Original code:

```python
import warnings


def slow_fib(n):
    if n > 10:
        warnings.warn("This Fibonacci function is slow")
    if n == 0 or n == 1:
        return n
    return slow_fib(n - 1) + slow_fib(n - 2)


def faster_fib(n):
    fibs = [0, 1, 1]
    while len(fibs) <= n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs[n]
```

Typed and documented code:

```python
"""Module to compute Fibonacci sequences."""

from typing import List
import warnings


def slow_fib(n: int) -> int:
    """Compute Fibonnaci sequence in a slow manner."""
    if n > 10:
        warnings.warn("This Fibonacci function is slow")
    if n == 0 or n == 1:
        return n
    return slow_fib(n - 1) + slow_fib(n - 2)


def faster_fib(n: int) -> int:
    """Compute Fibonnaci sequence in a faster manner.

    Uses an array, so requires more space.
    """
    fibs: List[int] = [0, 1, 1]
    while len(fibs) <= n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs[n]
```

Original code:

```python
{code}
```

Typed and documented code:

```python
'''.lstrip()
)

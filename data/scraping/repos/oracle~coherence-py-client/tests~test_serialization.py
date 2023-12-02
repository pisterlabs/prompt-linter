# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl.

from decimal import Decimal
from typing import Any

from coherence.serialization import JSONSerializer, proxy
from tests.Task import Task


def test_python_decimal() -> None:
    _verify_round_trip(Decimal("12.1345797249237923423872493"), True)


def test_python_large_integer() -> None:
    _verify_round_trip(9223372036854775810, True)  # larger than Java Long (2^63 - 1)


def test_python_large_negative_integer() -> None:
    _verify_round_trip(-9223372036854775810, True)  # less than Java Long -2^63


def test_python_java_long_upper_bound() -> None:
    _verify_round_trip(9223372036854775807, False)  # Java Long (2^32 - 1)


def test_python_java_long_lower_bound() -> None:
    _verify_round_trip(-9223372036854775809, False)  # Java Long (2^32 - 1)


def test_custom_object() -> None:
    _verify_round_trip(Task("Milk, eggs, and bread"), True)


def test_python_numerics_in_object() -> None:
    _verify_round_trip(Simple(), False)


@proxy("test.Simple")
class Simple:
    def __init__(self) -> None:
        super().__init__()
        self.n1 = (2**63) - 1
        self.n2 = self.n1 + 5
        self.n3 = Decimal("12.1345797249237923423872493")

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Simple):
            return self.n1 == getattr(o, "n1") and self.n2 == getattr(o, "n2") and self.n3 == getattr(o, "n3")

        return False


def _verify_round_trip(obj: Any, should_have_class: bool) -> None:
    serializer: JSONSerializer = JSONSerializer()
    ser_result: bytes = serializer.serialize(obj)
    print(f"Serialized [{type(obj)}] result: {ser_result.decode()}")

    if should_have_class:
        assert "@class" in ser_result.decode()

    deser_result: Any = serializer.deserialize(ser_result)
    print(f"Deserialized [{type(deser_result)}] result: {deser_result}")
    assert deser_result == obj

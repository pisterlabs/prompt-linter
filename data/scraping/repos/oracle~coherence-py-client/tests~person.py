# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl.

from __future__ import annotations

from coherence import NamedMap
from tests.address import Address


class Person:
    def __init__(self, name: str, gender: str, age: int, weight: float, address: Address, sports: list[str]):
        self.name = name
        self.gender = gender
        self.age = age
        self.weight = weight
        self.address = address
        self.sports = sports

    def __str__(self) -> str:
        return str(self.__dict__)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Person):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    @classmethod
    async def populate_named_map(cls, cache: NamedMap[str, Person]) -> None:
        await cache.put(Person.pat().name, Person.pat())
        await cache.put(Person.paula().name, Person.paula())
        await cache.put(Person.andy().name, Person.andy())
        await cache.put(Person.alice().name, Person.alice())
        await cache.put(Person.jim().name, Person.jim())
        await cache.put(Person.fred().name, Person.fred())
        await cache.put(Person.fiona().name, Person.fiona())

    @classmethod
    def create_person(
        cls, name: str, gender: str, age: int, weight: float, address: Address, sports: list[str]
    ) -> Person:
        return Person(name, gender, age, weight, address, sports)

    @classmethod
    def fred(cls) -> Person:
        addr = Address.address("1597 Olive Street", "San Francisco", "CA", 94102, "USA")
        return cls.create_person("Fred", "Male", 58, 185.5, addr, ["soccer", "tennis", "cricket"])

    @classmethod
    def fiona(cls) -> Person:
        addr = Address.address("2382 Palm Ave", "Daly City", "CA", 94014, "USA")
        return cls.create_person("Fiona", "Female", 29, 118.5, addr, ["tennis", "hiking"])

    @classmethod
    def pat(cls) -> Person:
        addr = Address.address("2038 Helford Lane", "Carmel", "IN", 46032, "USA")
        return cls.create_person("Pat", "Male", 62, 205.0, addr, ["golf", "pool", "curling"])

    @classmethod
    def paula(cls) -> Person:
        addr = Address.address("4218 Daniel St", "Champaign", "IL", 61820, "USA")
        return cls.create_person("Paula", "Female", 35, 125.0, addr, ["swimming", "golf", "skiing"])

    @classmethod
    def andy(cls) -> Person:
        addr = Address.address("1228 West Ave", "Miami", "FL", 33139, "USA")
        return cls.create_person("Andy", "Male", 25, 155.0, addr, ["soccer", "triathlon", "tennis"])

    @classmethod
    def alice(cls) -> Person:
        addr = Address.address("2208 4th Ave", "Phoenix", "AZ", 85003, "USA")
        return cls.create_person("Alice", "Female", 22, 110.0, addr, ["golf", "running", "tennis"])

    @classmethod
    def jim(cls) -> Person:
        addr = Address.address("37 Bowdoin St", "Boston", "MA", 2114, "USA")
        return cls.create_person("Jim", "Male", 36, 175.5, addr, ["golf", "football", "badminton"])

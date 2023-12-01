# Copyright (c) 2023, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl.

import asyncio
from dataclasses import dataclass
from typing import List

from coherence import Filters, NamedMap, Session
from coherence.filter import Filter


@dataclass
class Hobbit:
    """
    A simple class representing a Hobbit.
    """

    id: int
    name: str
    age: int
    home: str


async def do_run() -> None:
    """
    Demonstrates various Filter operations against a NamedMap.

    :return: None
    """
    session: Session = await Session.create()
    try:
        homes: List[str] = ["Hobbiton", "Buckland", "Frogmorton", "Stock"]
        named_map: NamedMap[int, Hobbit] = await session.get_map("hobbits")

        await named_map.clear()

        num_hobbits: int = 20
        print("Adding", num_hobbits, "random Hobbits ...")
        for i in range(num_hobbits):
            await named_map.put(i, Hobbit(i, "Hobbit-" + str(i), 15 + i, homes[i % 4]))

        print("NamedMap size is :", await named_map.size())

        print("Retrieve the Hobbits between the ages of 17 and 21 ...")
        async for entry in named_map.entries(Filters.between("age", 17, 21)):
            print("Key :", entry.key, ", Value :", entry.value)

        print("Retrieve the Hobbits between the ages of 17 and 30 and live in Hobbiton ...")
        query_filter: Filter = Filters.between("age", 17, 30).And(Filters.equals("home", "Hobbiton"))
        async for entry in named_map.entries(query_filter):
            print("Key :", entry.key, ", Value :", entry.value)

        print("Retrieve the Hobbits between the ages of 17 and 25 who live in Hobbiton or Frogmorton")
        query_filter = Filters.between("age", 17, 25).And(Filters.is_in("home", {"Hobbiton", "Frogmorton"}))
        async for entry in named_map.entries(query_filter):
            print("Key :", entry.key, ", Value :", entry.value)

    finally:
        await session.close()


asyncio.run(do_run())

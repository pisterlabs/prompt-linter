# Copyright (c) 2023, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl.

import asyncio
from dataclasses import dataclass
from typing import List

from coherence import NamedMap, Processors, Session


@dataclass
class Hobbit:
    """
    A simple class representing a Hobbit.
    """

    id: int
    name: str
    age: int


async def do_run() -> None:
    """
    Demonstrates various EntryProcessor operations against a NamedMap.

    :return: None
    """
    session: Session = await Session.create()
    try:
        named_map: NamedMap[int, Hobbit] = await session.get_map("hobbits")

        await named_map.clear()

        hobbit: Hobbit = Hobbit(1, "Bilbo", 111)
        print("Add new hobbit :", hobbit)
        await named_map.put(hobbit.id, hobbit)

        print("NamedMap size is :", await named_map.size())

        print("Hobbit from get() :", await named_map.get(hobbit.id))

        print("Update Hobbit using processor ...")
        await named_map.invoke(hobbit.id, Processors.update("age", 112))

        print("Updated Hobbit is :", await named_map.get(hobbit.id))

        hobbit2: Hobbit = Hobbit(2, "Frodo", 50)

        print("Add new hobbit :", hobbit2)
        await named_map.put(hobbit2.id, hobbit2)

        print("NamedMap size is :", await named_map.size())

        print("Sending all Hobbits ten years into the future!")
        keys: List[int] = []
        async for entry in named_map.invoke_all(Processors.increment("age", 10)):
            keys.append(entry.key)
            print("Updated age of Hobbit with id ", entry.key, "to", entry.value)

        print("Displaying all updated Hobbits ...")
        async for result in named_map.get_all(set(keys)):
            print(result.value)

        await named_map.remove(hobbit.id)
        await named_map.remove(hobbit2.id)
    finally:
        await session.close()


asyncio.run(do_run())

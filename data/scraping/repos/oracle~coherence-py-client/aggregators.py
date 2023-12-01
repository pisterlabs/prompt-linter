# Copyright (c) 2023, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl.

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import List

from coherence import Aggregators, Filters, NamedMap, Session


@dataclass
class Hobbit:
    """
    A simple class representing a Hobbit.
    """

    id: int
    name: str
    age: int
    hobbies: str


async def do_run() -> None:
    """
    Demonstrates various Aggregator operations against a NamedMap.

    :return: None
    """
    person_data = {
        1: Hobbit(1, "Bilbo", 111, "Burglaring"),
        2: Hobbit(2, "Frodo", 50, "Bearing"),
        3: Hobbit(3, "Sam", 38, "Side Kick"),
        4: Hobbit(3, "Meriadoc", 36, "Side Kick"),
        5: Hobbit(3, "Peregrin", 28, "Side Kick"),
    }

    session: Session = await Session.create()
    try:
        named_map: NamedMap[int, Hobbit] = await session.get_map("aggregation-test")

        await named_map.clear()

        await named_map.put_all(person_data)

        distinct_hobbies: List[str] = await named_map.aggregate(Aggregators.distinct("hobbies"))
        print("Distinct hobbies :", distinct_hobbies)

        count: int = await named_map.aggregate(Aggregators.count())
        print("Number of Hobbits :", count)

        over_forty: int = await named_map.aggregate(Aggregators.count(), filter=Filters.greater("age", 40))
        print("Number of Hobbits older than 40 :", over_forty)

        avg_under_forty: Decimal = await named_map.aggregate(Aggregators.average("age"), filter=Filters.less("age", 40))
        print("Average age of Hobbits under 40 :", int(avg_under_forty))

        print("The oldest Hobbit for each hobby ...")
        results: dict[str, int] = await named_map.aggregate(Aggregators.group_by("hobbies", Aggregators.max("age")))
        for hobby, age in results.items():
            print("Hobby: ", hobby, "Max age: ", age)
    finally:
        await session.close()


asyncio.run(do_run())

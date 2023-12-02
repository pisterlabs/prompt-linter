# Copyright (c) 2023, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl.

import asyncio

from coherence import Filters, NamedMap, Session
from coherence.event import MapLifecycleEvent, MapListener
from coherence.filter import MapEventFilter


async def do_run() -> None:
    """
    Demonstrates listeners for entry events and cache lifecycle.

    :return: None
    """
    session: Session = await Session.create()
    try:
        named_map: NamedMap[int, str] = await session.get_map("listeners-map")
        await named_map.put(1, "1")

        print("NamedMap lifecycle events")

        named_map.on(MapLifecycleEvent.RELEASED, lambda x: print("RELEASED", x))
        named_map.on(MapLifecycleEvent.TRUNCATED, lambda x: print("TRUNCATE", x))
        named_map.on(MapLifecycleEvent.DESTROYED, lambda x: print("DESTROYED", x))

        print("Truncating the NamedMap; this should generate an event ...")
        await named_map.truncate()
        await asyncio.sleep(1)

        print("Releasing the NamedMap; this should generate an event ...")
        named_map.release()
        await asyncio.sleep(1)

        print("Destroying the NamedMap; this should generate an event ...")
        await named_map.destroy()
        await asyncio.sleep(1)

        print("\n\nNamedMap entry events")

        named_map = await session.get_map("listeners-map")

        listener1: MapListener[int, str] = MapListener()
        listener1.on_any(lambda e: print(e))

        print("Added listener for all events")
        print("Events will be generated when an entry is inserted, updated, and removed")
        await named_map.add_map_listener(listener1)

        await named_map.put(1, "1")
        await named_map.put(1, "2")
        await named_map.remove(1)
        await asyncio.sleep(1)

        await named_map.remove_map_listener(listener1)

        print("\nAdded listener for all entries, but only when they are inserted")
        ins_filter = Filters.event(Filters.always(), MapEventFilter.INSERTED)
        await named_map.add_map_listener(listener1, ins_filter)

        await named_map.put(1, "1")
        await named_map.put(1, "2")
        await named_map.remove(1)
        await asyncio.sleep(1)

        await named_map.remove_map_listener(listener1, ins_filter)

        print("\nAdded listener for entries with a length larger than one, but only when they are updated or removed")
        upd_del_filter = Filters.event(Filters.greater("length()", 1), MapEventFilter.UPDATED | MapEventFilter.DELETED)
        await named_map.add_map_listener(listener1, upd_del_filter)

        for i in range(12):
            await named_map.put(i, str(i))
            await named_map.put(i, str(i + 1))
            await named_map.remove(i)

        await asyncio.sleep(1)

        await named_map.remove_map_listener(listener1, upd_del_filter)
        await named_map.clear()
    finally:
        await session.close()


asyncio.run(do_run())

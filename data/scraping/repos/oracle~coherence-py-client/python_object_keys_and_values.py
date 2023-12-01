# Copyright (c) 2023, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl.

import asyncio
from dataclasses import dataclass

from coherence import NamedMap, Processors, Session


@dataclass
class AccountKey:
    account_id: int
    account_type: str


@dataclass
class Account:
    account_id: int
    account_type: str
    name: str
    balance: float


async def do_run() -> None:
    """
    Demonstrates basic CRUD operations against a NamedMap using
    `AccountKey` keys with `Account` values.

    :return: None
    """
    session: Session = await Session.create()
    try:
        named_map: NamedMap[AccountKey, Account] = await session.get_map("accounts")

        await named_map.clear()

        new_account_key: AccountKey = AccountKey(100, "savings")
        new_account: Account = Account(new_account_key.account_id, new_account_key.account_type, "John Doe", 100000.00)

        print(f"Add new account {new_account} with key {new_account_key}")
        await named_map.put(new_account_key, new_account)

        print("NamedMap size is :", await named_map.size())

        print("Account from get() :", await named_map.get(new_account_key))

        print("Update account balance using processor ...")
        await named_map.invoke(new_account_key, Processors.update("balance", new_account.balance + 1000))

        print("Updated account is :", await named_map.get(new_account_key))

        await named_map.remove(new_account_key)

        print("NamedMap size is :", await named_map.size())
    finally:
        await session.close()


asyncio.run(do_run())

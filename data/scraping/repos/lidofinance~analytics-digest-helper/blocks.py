import pandas as pd
from langchain.schema import HumanMessage, SystemMessage
from .prompts import (
    thread_prompt,
    block_append_prompt,
    st_eth_apr_prompt,
    stEthOnL2Bridges_prompt,
    tvl_prompt,
    netDepositGrowthLeaders_prompt,
    stEthToEth_prompt,
    dexLiquidityReserves_prompt,
    totalStEthInDeFi_prompt,
)
from datetime import datetime
from langchain.chat_models.openai import ChatOpenAI


class BlockWriter:
    def __init__(self, end_date: str, start_date: str):
        self.end_date = end_date
        self.start_date = start_date
        self.write_functions = {
            "stETHApr": self.write_stETHApr,
            "tvl": self.write_tvl,
            "netDepositGrowthLeaders": self.write_netDepositGrowthLeaders,
            "stEthToEth": self.write_stEthToEth,
            "dexLiquidityReserves": self.write_dexLiquidityReserves,
            "bridgeChange": self.write_bridgeChange,
            "totalStEthInDeFi": self.write_totalStEthInDeFi,
        }

    def write_block(self, processed_input: str, system_prompt: str) -> str:
        today = datetime.today().strftime("%B %d %Y")

        chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")  # type: ignore
        thread = chat.predict_messages(
            [
                SystemMessage(content=system_prompt.format(DATE=today) + "\n" + block_append_prompt),
                HumanMessage(content=processed_input),
            ]
        )

        return thread.content

    def write_stETHApr(self, processed):
        return self.write_block(processed, st_eth_apr_prompt)

    def write_tvl(self, processed):
        return self.write_block(processed, tvl_prompt)

    def write_netDepositGrowthLeaders(self, processed):
        return self.write_block(processed, netDepositGrowthLeaders_prompt)

    def write_stEthToEth(self, processed):
        return self.write_block(processed, stEthToEth_prompt)

    def write_dexLiquidityReserves(self, processed):
        print(processed)
        return self.write_block(processed, dexLiquidityReserves_prompt)

    def write_bridgeChange(self, processed_bridge_change):
        return self.write_block(processed_bridge_change, stEthOnL2Bridges_prompt)

    def write_totalStEthInDeFi(self, processed):
        print(processed)
        return self.write_block(processed, totalStEthInDeFi_prompt)

    def compose_thread(self, processed: dict[str, str]):
        print(processed)

        processed_data = ""

        for k, v in processed.items():
                if self.write_functions.get(k) is None:
                    continue
                block = self.write_functions[k](v)
                print(block)
                processed_data += block + "\n\n"

        print(processed_data)
        chat = ChatOpenAI(temperature=0, model="gpt-4")  # type: ignore
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d %H:%M:%S")
        day_after_end_date = end_date + pd.DateOffset(days=1)
        end_date_str = day_after_end_date.strftime("%B %d %Y")
        thread = chat.predict_messages(
            [
                SystemMessage(content=thread_prompt.format(start_date=self.start_date, end_date=end_date_str)),
                HumanMessage(content=processed_data),
            ]
        )

        return thread.content

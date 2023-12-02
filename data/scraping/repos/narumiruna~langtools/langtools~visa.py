from typing import Optional
from typing import Type

import visafx
from langchain.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field


class ExchangeRateCalculatorInput(BaseModel):
    amount: float = Field(description='The amount to convert.')
    from_curr: str = Field(description='The currency to convert from.')
    to_curr: str = Field(description='The currency to convert to.')


class ExchangeRateCalculator(BaseTool):
    """"Exchange Rate Calculator

    Use the converter below to get an indication of the rate you may receive
    when using your Visa card to pay while traveling internationally.
    """

    name = "calculate_exchange_rate"
    description = ('A Visa exchange rate calculator.'
                   'Input should be an amount, a currency to convert from, and a currency to convert to. '
                   'The output will be the converted amount and the exchange rate.')

    args_schema: Optional[Type[BaseModel]] = ExchangeRateCalculatorInput

    def _run(self, amount: str, from_curr: str, to_curr: str) -> str:
        # the from_curr and to_curr are reversed in the API
        r = visafx.rates(amount, from_curr=to_curr, to_curr=from_curr)
        return (f'Amount: {amount}\n'
                f'From: {from_curr}\n'
                f'To: {to_curr}\n'
                f'Converted amount: {r.convertedAmount}\n'
                f'Exchange rate: {r.fxRateWithAdditionalFee}\n')

    async def _arun(self, amount: str, from_curr: str, to_curr: str) -> str:
        return self._run(amount, from_curr, to_curr)

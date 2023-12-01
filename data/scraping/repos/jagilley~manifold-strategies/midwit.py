from .strategy import Strategy
from datetime import datetime, timedelta
from manifoldpy.manifoldpy.api import Market
from .time_utils import get_price_at_timestamp, get_price
import openai
import json

class Midwit(Strategy):
    """A strategy that uses GPT to determine whether a market will resolve to YES or NO"""
    def __init__(self):
        super().__init__("Midwit")

    def strategy(self, market: Market):
        created_datetime = datetime.fromtimestamp(market.createdTime / 1000)
        one_day_after = created_datetime + timedelta(hours=6)
        one_day_after_timestamp = one_day_after.timestamp()
        one_day_after_timestamp_ms = one_day_after_timestamp * 1000

        price_at_one_day_after = get_price_at_timestamp(market, one_day_after_timestamp_ms)
        current_price = get_price_at_timestamp(market, datetime.now().timestamp() * 1000)

        if current_price - price_at_one_day_after == 0:
            return None, None

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Determine whether this market will resolve to YES or NO using your best judgement:" + market.question},
            ],
            functions=[
                {
                    "name": "yes_or_no",
                    "description": "Returns either YES or NO",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "yes_or_no": {"type": "string", "enum": ["YES", "NO"]},
                        },
                        "required": ["yes_or_no"],
                    },
                }
            ],
            function_call={"name": "yes_or_no"}
        )

        # print(completion.choices[0])
        json_resp = completion.choices[0]['message']['function_call']['arguments']
        position = json.loads(json_resp)['yes_or_no']
        print(market.question, position)

        if position not in ['YES', 'NO']:
            return None, None

        delta = self.evaluate_strategy(price_at_one_day_after, current_price, position)
        return delta, position

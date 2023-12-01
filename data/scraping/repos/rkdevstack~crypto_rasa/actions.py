# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
import json, requests
from rasa_sdk import Action, Tracker, FormValidationAction, ValidationAction
from rasa_sdk.events import EventType
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet
from datetime import datetime, date, timedelta
import openai

import os
from twilio.rest import Client

from decimal import Decimal

#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

import csv

# csv fileused id Geeks.csv
filename1 = "D:\kuwait\crypto2\data\Coin List.csv"
filename2 = "D:\kuwait\crypto2\data\exchange.csv"
filename3 = "D:\kuwait\crypto2\data\Stocks List.csv"
# opening the file using "with"
# statement

polygonapikey ="ZRHS94oFWcLRUUttMeSubXp4iRt_42xI"
coinlist = dict()
with open(filename1, 'r', encoding='cp850') as data:
    for line in csv.reader(data):
        coinlist[line[0][1:]] = (line[1].lower(), line[2][1:].lower())

exchangelist= dict()
with open(filename2, 'r', encoding='cp850') as data:
    for line in csv.reader(data):
        exchangelist[line[1]]= line[0]

stocklist= dict()
with open(filename3, 'r', encoding='cp850') as data:
    for line in csv.reader(data):
        stocklist[line[0]]= line[1]


quotes_latest_v2={"num_market_pairs": "number of market pairs",
                  "circulating_supply": "circulating supply",
                  "total_supply": "total supply",
                  "max_supply": "maximum supply",
                  "cmc_rank":"current rank",
                  "is_active":"status",
                  "token_address":"token address",
                  "market_cap":"market cap",
                  "market_cap_dominance":"market cap dominance",
                  "fully_diluted_market_cap":"fully diluted market cap"}

global_metric_data={
    "total_cryptocurrencies":[["total_cryptocurrencies","active_cryptocurrencies"],"There are {} out of which {} are active"],
    "total_exchanges":[["total_exchanges" , "active_exchanges"],"There are {} out of which {} are active"],
    "defi_market_cap":[["defi_market_cap","defi_24h_percentage_change"],"The market cap of defi is {} with a percentage change of {} over the last day"],
    "defi_volume_24h_reported":[["defi_volume_24h_reported"],"The current trading volume of defi is {} "],
    "stablecoin_market_cap":[["stablecoin_market_cap","stablecoin_24h_percentage_change"],"The market cap of stablecoin is {} with a percentage change of {} over the last day"],
    "stablecoin_volume_24h_reported": [["stablecoin_volume_24h_reported"],"The current trading volume of stablecoin is {}"],
    "derivatives_volume_24h_reported": [["derivatives_volume_24h_reported"],"The current trading volume of derivatives is {}"],
    "total_market_cap":[["total_market_cap" ,"total_market_cap_yesterday_percentage_change"],"The total market cap of cryptocurrencies is {} with a percentage change of {} over the last trading day"],
    "total_volume_24h_reported":[["total_volume_24h_reported"],"The current trading volume of cryptocurrencies is {}"],
    "altcoin_volume_24h_reported":[["altcoin_volume_24h_reported"],"The current trading volume of altcoin is {}"],
    "altcoin_market_cap":[["altcoin_market_cap"],"The market cap of altcoin is {}"]

}

metadata_format={
    "symbol" : "The symbol for {} is {}",
    "name" : "The name for {} is {}",
    "description" : "{1}",
    "website" : "Please click the link below to visit the website of {}",
    "twitter" : "Please click the link below to visit the twitter account of {}",
    "message_board" : "Please click the link below to visit the forum of {}",
    "chat" : "Please click the link below to visit the chatting platform  of {}",
    "explorer" : "Please click the link below to visit the blockchain explorer  of {}",
    "reddit" : "Please click the link below to visit the reddit page of {}",
    "technical_doc" : "Please click the link below to access the white paper of {}",
    "source_code" : "Please click the link below to access the source code of {}"
}

exchange_var_data={
    "num_coins" : "The cryptocurrencies listed on {} is {}",
    "traffic_score" : "The traffic score of {} is {}",
    "exchange_score" : "The exchange score of {} is {}",
    "liquidity_score" : "The liquidity score of {} is {}",
    "volume_24h" : [["volume_24h","volume_change_24h"], "The current trading volume on {} is {} with a percentage change of {} over the last trading day"],
    "spot_volume_usd" : "The spot volume on {} is {}",
    "derivative_volume_usd" : "The derivative volume on {} is {}",
    "open_interest" : "The open interest on {} is {}"
}

exchange_fileds={"open_intrest":"open intrest"}


class ActionAskCryptoName2(Action):
    def name(self) -> Text:
        return "action_ask_crypto_name2"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        slug = tracker.get_slot("duplicate_coin")
        print(slug)
        print("---------AskForCryptoName2Action------------")
        if slug!=None and len(slug) != 0:

             dispatcher.utter_message(text=f"What kind of coin do you want?",buttons=[{"title": p,"payload":p} for p in slug[-1]])

             return []
        else:
            return [SlotSet("requested_slot", None)]
        # dispatcher.utter_message(text=f"What kind of coin do you want?",
        #                           buttons=[{"title": p, "payload": p} for p in slug])
        # return []

class ActionAskSymbol(Action):
    def name(self) -> Text:
        return "action_ask_symbol"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        stocks = tracker.get_slot("stocks")
        stocks = stocks.upper().strip('#')
        print("stocks",stocks)
        print("---------AskForCryptostockssymbol------------")
        duplicate_stocks=[]
        if stocks.upper() not in stocklist.keys():
            for i in stocklist:
                if stocklist[i].lower()== stocks.lower():
                    duplicate_stocks.append(i)

            if len(duplicate_stocks)>1:
                dispatcher.utter_message(text=f"What kind of stock do you want?",buttons=[{"title": p,"payload":p} for p in duplicate_stocks])
                #dispatcher.utter_message(template="utter_user_details")
                return []
            else:
                return [SlotSet("requested_slot", None),SlotSet("symbol", duplicate_stocks[0])]

        else:
            return [SlotSet("requested_slot", None),SlotSet("symbol", stocks)]
class ActionGetPrice(Action):

     def name(self) -> Text:
         return "action_get_price"


     @staticmethod
     def roundval(val):
        if val==None:
            return val
        elif type(val)==str:
            return val
        elif abs(val)  >= 1:
            return "{:,.2f}".format(float(val))
        elif 1 > abs(val)  >= 0.01:
            return "{:,.5f}".format(float(val))
        else:
            return "{:,.10f}".format(float(val))
     @staticmethod
     def timeframe(timeval):
        if timeval == "1h":
            return "1 hour"
        elif timeval == "24h":
            return "24 hours"
        elif timeval == "7d":
            return "7 days"
        elif timeval == "30d":
            return "30 days"
        elif timeval == "60d":
            return "60 days"
        elif timeval == "90d":
            return "90 days"
        else:
            return "all time"


     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

         #crypto_coin = next(tracker.get_latest_entity_values("crypto"), None)
         sluglist = tracker.get_slot("unique_coin")
         print("sluglist", sluglist)
         print("---------ActionGetPrice------------")
         duration = tracker.get_slot("duration")
         print(duration)
         all_time = tracker.get_slot("hl_price")
         print(all_time)
         price_change=tracker.get_slot("price_change")
         print("price_change",price_change)
         time = tracker.get_slot("time")
         print("time", time)

         if all_time:
             if duration is None:
                 duration="all_time"
             for slug in sluglist:
                url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/price-performance-stats/latest?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&slug={}&time_period={}'.format(slug, duration)
                r = json.loads(requests.get(url).content)['data']
                key = list(r.keys())
                high = json.loads(requests.get(url).content)['data'][key[0]]["periods"][duration]["quote"]["USD"]["high"]
                low = json.loads(requests.get(url).content)['data'][key[0]]["periods"][duration]["quote"]["USD"]["low"]
                dispatcher.utter_message(text="{} {} statics: \nHIGH: {}\nLOW: {}\n".format(coinlist[slug][1], self.timeframe(duration),self.roundval(high), self.roundval(low)))

         elif price_change:
             if duration is None:
                 duration = time
             for slug in sluglist:
                url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&slug={}'.format(slug)
                r = json.loads(requests.get(url).content)['data']
                key = list(r.keys())
                percent_change = r[key[0]]["quote"]["USD"]["percent_change_"+duration]
                price = r[key[0]]["quote"]["USD"]["price"]
                dispatcher.utter_message(text="{} is currently trading at ${} with a percentage change of {}% in the last {}".format(coinlist[slug][1], self.roundval(price),self.roundval(percent_change),self.timeframe(duration)))

         else:
             for slug in sluglist:
                url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&slug={}'.format(slug)
                r = json.loads(requests.get(url).content)['data']
                key = list(r.keys())
                r = r[key[0]]["quote"]["USD"]["price"]
                dispatcher.utter_message(text="The price of {} is {}".format(coinlist[slug][1],self.roundval(r)))

         return []

class ActionValidateCoin(Action):

    def name(self) -> Text:
        return "action_validate_Coin"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        #crypto_coin = next(tracker.get_latest_entity_values("crypto_dup"), None)
        crypto_list=tracker.get_slot("crypto_dup")
        crypto_coin=[]
        crypto_list=[s.strip('$') for s in crypto_list]
        for i in crypto_list:
            if i not in crypto_coin:
                crypto_coin.append(i)
        print(crypto_coin)
        if crypto_coin:
            slug = []
            print(crypto_coin)
            print("---------cryptonameonly1------------")
            for coin in crypto_coin:
                slug_dup=[]
                for i in coinlist:
                    if coin.lower() in coinlist[i]:
                        slug_dup.append(i)
                slug.append(slug_dup)
            while [] in slug:
                slug.remove([])
            print(slug)
            unique_coin=[]
            duplicate_coin=[]
            for i in slug:
                if len(i) ==1:
                    unique_coin.append(i[0])
                else:
                    duplicate_coin.append(i)

            if (len(slug) == len(unique_coin)):
                print("yes")
                return [ SlotSet("requested_slot", None), SlotSet("unique_coin",unique_coin)]
            else:
                print("no")
                return [SlotSet("unique_coin",unique_coin),SlotSet("duplicate_coin",duplicate_coin)]
        elif (tracker.get_slot("crypto_name2")):
            return [SlotSet("requested_slot", None)]
        else:
            return []

class ActionValidateSlots(Action):

    def name(self) -> Text:
        return "action_validate_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            duration = list(filter(lambda entity: entity['entity'] == "duration", tracker.latest_message["entities"]))
            #print(duration)
            print("+++++++++data++++++++++++++++")
            durationslot=None
            time=list(filter(lambda entity: entity['entity'] == "time", tracker.latest_message["entities"]))
            if duration:
                values = duration[0]["additional_info"]["value"]
                unit = duration[0]["additional_info"]["unit"]
                print(f'{values}{unit}')
                print("+++++++++duration++++++++++++++++")
                if (unit=="day"):
                    unit="d"
                if (unit=="hour"):
                    unit="h"
                if (unit=="month"):
                    values=values*30
                    unit="d"
                if (unit=="week"):
                    values=values*7
                    unit="d"
                durationslot = f'{values}{unit}'
            print(durationslot)

            #print(time)
            timeslot=None
            if time:
                date1 = time[0]["additional_info"]["values"][0]["to"]["value"]
                date2 = time[0]["additional_info"]["values"][0]["from"]["value"]
                print("---- date ----")
                unit = time[0]["additional_info"]["values"][0]["to"]["grain"]
                print(unit)
                print("----------unit-----")
                if (unit == "day"):
                    unit = "d"
                if (unit == "hour"):
                    unit = "h"
                date_format = "%Y-%m-%d"
                a = datetime.strptime(date1[:10], date_format)
                b = datetime.strptime(date2[:10], date_format)
                delta = abs(b - a)
                values = delta.days
                if values==1:
                    values=24
                if values == 0:
                    start_time = datetime.strptime(date1[11:19], "%H:%M:%S")
                    end_time = datetime.strptime(date2[11:19], "%H:%M:%S")
                    delta = end_time - start_time
                    sec = delta.total_seconds()
                    values = int(abs(sec / (60 * 60)))
                    print(values)
                timeslot = f'{values}{unit}'
                print(timeslot)
                # validation failed, set this slot to None

            return [SlotSet("duration", durationslot),SlotSet("time", timeslot)]

class ActionGetVolume(Action):

    def name(self) -> Text:
        return "action_get_volume"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val)  >= 1:
            return "{:,.2f}".format(float(val))
        elif 1 > abs(val)  >= 0.01:
            return "{:,.5f}".format(float(val))
        else:
            return "{:,.10f}".format(float(val))
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        #crypto_coin = next(tracker.get_latest_entity_values("crypto"), None)
        sluglist = tracker.get_slot("unique_coin")
        for slug in sluglist:
            print("---------ActionGetVolume------------")
            url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&slug={}'.format(slug)
            r = json.loads(requests.get(url).content)['data']
            key = list(r.keys())
            r = r[key[0]]["quote"]["USD"]["volume_24h"]
            dispatcher.utter_message(text="The volume of {} is {}".format(coinlist[slug][1], self.roundval(r)))

        return []

class ValidateFancyCryptoForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_fancy_crypto_form"

    def validate_crypto_name2(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `crypto_name` value."""

        duplicate_coin = tracker.get_slot("duplicate_coin")
        duplicate_coin=duplicate_coin[:-1]
        print("duplicate_coin", duplicate_coin)
        unique_coin = tracker.get_slot("unique_coin")
        print("unique_coin_Before", unique_coin)
        unique_coin.insert(0,slot_value.lower())
        print("unique_coin", unique_coin)

        if len(duplicate_coin)>0:
            return {"crypto_name2": None,"unique_coin": unique_coin,"duplicate_coin":duplicate_coin}
        # dispatcher.utter_message(text=f"OK! You want to have a {slot_value} price.")
        print(slot_value)
        print("---------ValidateFancyCryptoForm------------")
        return {"crypto_name2": slot_value,"unique_coin": unique_coin,"duplicate_coin":duplicate_coin}

class ValidateStocksForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_stocks_form"

    def validate_symbol(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `crypto_name` value."""
        # dispatcher.utter_message(text=f"OK! You want to have a {slot_value} price.")
        print("---------ValidateStocksform------------")
        return {"symbol": slot_value}
class ResetSlot(Action):

    def name(self):
        return "action_reset_slot"

    def run(self, dispatcher, tracker, domain):
        return [SlotSet("hl_price", None), SlotSet("duration", None), SlotSet("time", None), SlotSet("price_change", None), SlotSet("crypto_name2", None), SlotSet("symbol", None),SlotSet("stocks", None)]

class ValidatePredefinedSlots(ValidationAction):

    def validate_duration(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate location value."""

        #data = tracker.latest_message["entities"]
        duration = list(filter(lambda entity: entity['entity'] == "duration", tracker.latest_message["entities"] ))
        print(duration)
        print("+++++++++data++++++++++++++++")
        if duration:
            values = duration[0]["additional_info"]["value"]
            unit = duration[0]["additional_info"]["unit"]
            print(f'{values}{unit}')
            print("+++++++++duration++++++++++++++++")
            return {"duration": f'{values}{unit}'}
        else:
            # validation failed, set this slot to None
            return {"duration": None}


class ActionGetTopGainersLosers(Action):

    def name(self) -> Text:
        return "action_get_top_gainers_losers"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val) >= 1:
            return "{:,.2f}".format(float(val))
        elif 1 > abs(val) >= 0.01:
            return "{:,.5f}".format(float(val))
        else:
            return "{:,.10f}".format(float(val))

    @staticmethod
    def timeframe(timeval):
        if timeval == "1h":
            return "1 hour"
        elif timeval == "24h":
            return "24 hours"
        elif timeval == "7d":
            return "7 days"
        elif timeval == "30d":
            return "30 days"
        elif timeval == "60d":
            return "60 days"
        elif timeval == "90d":
            return "90 days"
        else:
            return ""

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        top_gl = tracker.get_slot("top_gl")
        duration = tracker.get_slot("duration")
        time = tracker.get_slot("time")
        limit=5
        if top_gl.lower() == "top gainers":
            order="desc"
        else:
            order="asc"
        if duration is None:
            duration = time
        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/trending/gainers-losers?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&limit={}&sort_dir={}&time_period={}'.format(limit,order,duration)
        r = json.loads(requests.get(url).content)['data']
        coins = [r[i]['name'] for i in range(len(r))]
        price = [self.roundval(r[i]['quote']['USD']['price']) for i in range(len(r))]
        change = [self.roundval(r[i]['quote']['USD']['percent_change_'+duration]) for i in range(len(r))]
        data= "The {} during the past {} are: \n \n".format(top_gl,self.timeframe(duration))
        for i in range(limit):
            data=data+"{}. {} currently trading at ${} gained {}% \n".format(i + 1, coins[i], self.roundval(price[i]), self.roundval(change[i]))
        dispatcher.utter_message(text=data)
        return []

class ActionGetMostVisited(Action):

    def name(self) -> Text:
        return "action_get_most_visited"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val)  >= 1:
            return "{:,.2f}".format(float(val))
        elif 1 > abs(val)  >= 0.01:
            return "{:,.5f}".format(float(val))
        else:
            return "{:,.10f}".format(float(val))

    @staticmethod
    def timeframe(timeval):
        if timeval == "1h":
            return "1 hour"
        elif timeval == "24h":
            return "24 hours"
        elif timeval == "7d":
            return "7 days"
        elif timeval == "30d":
            return "30 days"
        elif timeval == "60d":
            return "60 days"
        elif timeval == "90d":
            return "90 days"
        elif timeval == "365d":
            return "365 days"
        else:
            return ""

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        duration = tracker.get_slot("duration")
        time = tracker.get_slot("time")
        limit=5
        if duration is None:
            duration = time
        if duration not in ["24h","30d","7d"]:
            duration = "24h"
        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/trending/most-visited?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&limit={}&time_period={}'.format(limit,duration)
        r = json.loads(requests.get(url).content)['data']
        coins = [r[i]['name'] for i in range(len(r))]
        price = [self.roundval(r[i]['quote']['USD']['price']) for i in range(len(r))]
        change = [self.roundval(r[i]['quote']['USD']['percent_change_'+duration]) for i in range(len(r))]
        data="The most visited coins during the past {} are:\n \n".format(self.timeframe(duration))
        for i in range(limit):
            data = data + "{}. {} currently trading at ${} gained {}% \n".format(i + 1, coins[i], self.roundval(price[i]), self.roundval(change[i]))
        dispatcher.utter_message(text=data)
        return []

class ActionGetNewlyAddedCoins(Action):

    def name(self) -> Text:
        return "action_get_new_coins"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val) >= 1:
            val = "{:,.2f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val
        elif 1 > abs(val) >= 0.01:
            val = "{:,.5f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val
        else:
            val = "{:,.10f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        #duration = tracker.get_slot("duration")
        #time = tracker.get_slot("time")
        limit=5
        #if duration is None:
            #duration = time
        #if duration not in ["24h","30d","7d"]:
            #duration = "24h"
        print("------Newly--Added--Coins")
        duration="24h"
        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/new?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&limit={}'.format(limit)
        r = json.loads(requests.get(url).content)['data']
        #print(r)
        coins = [r[i]['name'] for i in range(len(r))]
        price = [self.roundval(r[i]['quote']['USD']['price']) for i in range(len(r))]
        change = [self.roundval(r[i]['quote']['USD']['percent_change_'+duration]) for i in range(len(r))]
        data = "Below are some of the new cryptocurrencies added :\n \n"
        for i in range(limit):
            data = data + "{}. {} currently trading at ${} gained {}% \n".format(i + 1, coins[i], self.roundval(price[i]), self.roundval(change[i]))
        dispatcher.utter_message(text=data)
        return []


class ActionGetTrendingLatest(Action):

    def name(self) -> Text:
        return "action_get_trending_latest"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val) >= 1:
            val = "{:,.2f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val
        elif 1 > abs(val) >= 0.01:
            val = "{:,.5f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val
        else:
            val = "{:,.10f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val

    @staticmethod
    def timeframe(timeval):
        if timeval == "1h":
            return "1 hour"
        elif timeval == "24h":
            return "24 hours"
        elif timeval == "7d":
            return "7 days"
        elif timeval == "30d":
            return "30 days"
        elif timeval == "60d":
            return "60 days"
        elif timeval == "90d":
            return "90 days"
        else:
            return ""

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        duration = tracker.get_slot("duration")
        time = tracker.get_slot("time")
        limit=5
        if duration is None:
            duration = time
        if duration not in ["24h","30d","7d"]:
            duration = "24h"
        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/trending/latest?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&limit={}&time_period={}'.format(limit,duration)
        r = json.loads(requests.get(url).content)['data']
        coins = [r[i]['name'] for i in range(len(r))]
        price = [self.roundval(r[i]['quote']['USD']['price']) for i in range(len(r))]
        change = [self.roundval(r[i]['quote']['USD']['percent_change_'+duration]) for i in range(len(r))]
        data="The most searched coins during the past {} are: \n \n".format(self.timeframe(duration))
        for i in range(limit):
            data = data + "{}. {} currently trading at ${} with a percentage change of {}% \n".format(i + 1, coins[i], self.roundval(price[i]), self.roundval(change[i]))
        dispatcher.utter_message(text=data)
        return []

class ActionGetExtraQuestions(Action):

    def name(self) -> Text:
        return "action_get_extra_questions"

    @staticmethod
    def roundval(val):
        if val==None:
            return val
        elif type(val) ==str:
            return val
        elif abs(val) >= 1:
            val="{:,.2f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val
        elif 1 > abs(val) >= 0.01:
            val = "{:,.5f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val
        else:
            val = "{:,.10f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # crypto_coin = next(tracker.get_latest_entity_values("crypto"), None)
        sluglist = tracker.get_slot("unique_coin")
        for slugval in sluglist:
            print(slugval)
            aux = tracker.get_slot("extra_q")
            print("aux",aux)
            print("---------ActionGetExtraQuestions------------")
            url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&slug={}&aux=num_market_pairs,cmc_rank,date_added,tags,platform,max_supply,circulating_supply,total_supply,market_cap_by_total_supply,volume_24h_reported,volume_7d,volume_7d_reported,volume_30d,volume_30d_reported,is_active,is_fiat".format(slugval)
            r = json.loads(requests.get(url).content)["data"]
            key = list(r.keys())
            if aux in r[key[0]]:
                res=r[key[0]][aux]
            elif aux in r[key[0]]["quote"]["USD"]:
                res = r[key[0]]["quote"]["USD"][aux]
            elif aux in r[key[0]]["platform"]:
                res = r[key[0]]["platform"][aux]
            if aux=="is_active":
                if int(res)==1:
                    res="active"
                else:
                    res="not active"
            dispatcher.utter_message(text="The {} of {} is {}".format(quotes_latest_v2[aux],coinlist[slugval][1], self.roundval(res)))

        return []
class ActionGetGlobalMetrics(Action):

    def name(self) -> Text:
        return "action_get_global_metrics"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val) >= 1:
            val = "{:,.2f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val
        elif 1 > abs(val) >= 0.01:
            val = "{:,.5f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val
        else:
            val = "{:,.10f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val


    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        global_metrics = tracker.get_slot("global_metrics")
        print("global_metrics",global_metrics)
        if global_metrics == "num_coins":
            global_metrics = "total_cryptocurrencies"
        if global_metrics == "derivative_volume_usd":
            global_metrics = "derivatives_volume_24h_reported"
        print("---------ActionGetGlobalMetrics------------")
        url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc"
        r = json.loads(requests.get(url).content)["data"]
        data=[]
        if global_metrics in r.keys():
            for i in global_metric_data[global_metrics][0]:
                data.append(r[i])
        elif global_metrics in r["quote"]["USD"]:
            for i in global_metric_data[global_metrics][0]:
                data.append(r["quote"]["USD"][i])
        if len(global_metric_data[global_metrics][0])==2:
            dispatcher.utter_message(text=global_metric_data[global_metrics][-1].format(self.roundval(data[0]),self.roundval(data[1])))
        elif len(global_metric_data[global_metrics][0])==1:
            dispatcher.utter_message(text=global_metric_data[global_metrics][-1].format(self.roundval(data[0])))
        return []

class ActionGetMetadata(Action):

    def name(self) -> Text:
        return "action_get_metadata"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        sluglist = tracker.get_slot("unique_coin")
        metadata = tracker.get_slot("metadata")
        print("metadata", metadata)
        for slugval in sluglist:
            print("---------ActionGetmetadata------------")
            url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/info?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&slug={}".format(slugval)
            r = json.loads(requests.get(url).content)["data"]
            # print(r)
            key = list(r.keys())
            if metadata in r[key[0]].keys():
                res = r[key[0]][metadata]
                if metadata=="name":
                    dispatcher.utter_message(text="The {} for {} is {}".format(metadata, r[key[0]]["symbol"], res))
                else:
                    dispatcher.utter_message(text=metadata_format[metadata].format(coinlist[slugval][1], res))
            elif metadata in r[key[0]]["urls"].keys():
                res=r[key[0]]["urls"][metadata]
                if len(res)>0:
                    dispatcher.utter_message(text=metadata_format[metadata].format(coinlist[slugval][1]))
                    for i in res:
                        dispatcher.utter_message(text="{}\n".format(i))
                else:
                    dispatcher.utter_message(text="{} option not available\n".format(metadata.replace('_', ' ')))
        return []

class ActionGetExchange(Action):

    def name(self) -> Text:
        return "action_get_exchange"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val) >= 1:
            val = "{:,.2f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val
        elif 1 > abs(val) >= 0.01:
            val = "{:,.5f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val
        else:
            val = "{:,.10f}".format(float(val))
            return val.rstrip('0').rstrip('.') if '.' in val else val

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        exchange = tracker.get_slot("exchange")
        print("exchange",exchange)
        #print(exchangelist)
        for i in exchangelist:
            if exchange.lower() == exchangelist[i]:
                exchange_slugval = i
                break
        exchange_var = tracker.get_slot("exchange_var")
        print("exchange_slugval",exchange_slugval)
        print("exchange_var", exchange_var)
        print("---------ActionGetexchange------------")
        url = "https://pro-api.coinmarketcap.com/v1/exchange/quotes/latest?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&slug={}".format(exchange_slugval)
        r = json.loads(requests.get(url).content)["data"][exchange_slugval]
        if exchange_var in r.keys():
            res = r[exchange_var]
            dispatcher.utter_message(text=exchange_var_data[exchange_var].format(exchange, self.roundval(res)))
        elif exchange_var in r["quote"]["USD"].keys():
            res = r["quote"]["USD"][exchange_var]
            if exchange_var =="volume_24h":
                dispatcher.utter_message(
                    text=exchange_var_data[exchange_var][1].format(exchange,self.roundval(res),self.roundval(r["quote"]["USD"]["percent_change_volume_24h"])))

            else:
                dispatcher.utter_message(
                    text=exchange_var_data[exchange_var].format(exchange, self.roundval(res)))

        return []

class ActionGetExchangeInfo(Action):

    def name(self) -> Text:
        return "action_get_exchange_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        slugval = tracker.get_slot("exchange")
        exchange_info = tracker.get_slot("exchange_info")
        print("exchange_info", exchange_info)
        print("---------ActionGetExchangemetadata------------")
        url = "https://pro-api.coinmarketcap.com/v1/exchange/info?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&slug={}".format(slugval)
        r = json.loads(requests.get(url).content)["data"][slugval]
        # print(r)
        if exchange_info in r.keys():
            res = r[exchange_info]
            if exchange_info == "date_launched":
                dispatcher.utter_message(text="{} was launched in {}".format(slugval, res[0:10]))
            elif exchange_info == "weekly_visits":
                dispatcher.utter_message(text="A total of {} people visited {} in the last 7 days".format(res, slugval))
            else:
                dispatcher.utter_message(text="The {} of {} is {}".format(exchange_info, slugval, res))

        elif exchange_info in r["urls"].keys():
            res = r["urls"][exchange_info]
            if len(res) > 0:
                dispatcher.utter_message(
                    text="For more information related to {} of {}, please visit the below link".format(exchange_info, slugval))
                for i in res:
                    dispatcher.utter_message(text="{}\n".format(i))
            else:
                dispatcher.utter_message(text="{} option not available\n".format(exchange_info))
        return []

class ActionGetExchangeMap(Action):

    def name(self) -> Text:
        return "action_get_exchange_map"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        slugval = tracker.get_slot("exchange")
        exchange_active = tracker.get_slot("exchange_active")
        print("exchange_active", exchange_active)
        print("---------ActionGetExchangemap------------")
        url = "https://pro-api.coinmarketcap.com/v1/exchange/map?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&slug={}".format(slugval)

        r = json.loads(requests.get(url).content)["data"][0]["is_active"]
        # print(r)
        if r == 1:
            dispatcher.utter_message(text="{} is currently active".format(slugval))
        else:
            dispatcher.utter_message(text="{} is currently not active".format(slugval))
        return []

class ActionGetPriceCoversion(Action):

    def name(self) -> Text:
        return "action_get_price_conversion"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val) >= 1:
            return "{:,.2f}".format(float(val))
        elif 1 > abs(val) >= 0.01:
            return "{:,.5f}".format(float(val))
        else:
            return "{:,.10f}".format(float(val))

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        #amount = list(filter(lambda entity: entity['entity'] == "number", tracker.latest_message["entities"]))
        amount = tracker.get_slot("number")
        if amount==None:
            amount=1
        sluglist=tracker.get_slot("unique_coin")
        symbol=[]
        coin_name=[]
        for i in sluglist:
            symbol.append(coinlist[i][0])
            coin_name.append(coinlist[i][1])
        print("coin_name",coin_name)
        print("symbol",symbol)
        print("amount", amount)
        print("---------ActionGetPriceCoversion------------")
        url = "https://pro-api.coinmarketcap.com/v2/tools/price-conversion?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&symbol={}&amount={}&convert={}".format(symbol[0],amount,','.join(symbol[1:]))
        r = json.loads(requests.get(url).content)["data"]
        for j in r:
            print(j)
            if j["name"].lower() == coin_name[0]:
                for i in symbol[1:]:
                    print(i)
                    price= j["quote"][i.upper()]["price"]
                    dispatcher.utter_message(text="{} {} = {} {}".format(amount, coin_name[0], self.roundval(price), i))
        return []

class ActionGetAirDrops(Action):

    def name(self) -> Text:
        return "action_get_air_drops"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val) >= 1:
            return "{:,.2f}".format(float(val))
        elif 1 > abs(val) >= 0.01:
            return "{:,.5f}".format(float(val))
        else:
            return "{:,.10f}".format(float(val))

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print("---------ActionGetairdrops------------")
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/airdrops?CMC_PRO_API_KEY=bdb69120-c6d3-4cd0-8777-23c11b947ddc&limit=5"
        r = json.loads(requests.get(url).content)["data"]
        data = "Below is the list of the ongoing airdrops happening of CoinMarketCap \n \n"
        for i in r:
            if i["coin"] != None:
                data = data + "{} - {} \n".format(i["coin"]["name"],i["link"])
        dispatcher.utter_message(text=data)
        return []
class ActionOutOfScope(Action):

    def name(self) -> Text:
        return "action_out_of_scope"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Set the API key
        openai.api_key = "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"
        # Use the ChatGPT model to generate text
        model_engine = "text-davinci-003"
        prompt = tracker.latest_message.get("text")
        print("prompt",prompt)
        completion = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=20, n=1, stop=None,
                                                  temperature=0.7)
        message = completion.choices[0].text
        print("chatGPT",message)
        dispatcher.utter_message(text=message)
        return []

class ActionGetTickerDetails(Action):

    def name(self) -> Text:
        return "action_get_ticker_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print("---------action_get_ticker_details------------")
        symbol = tracker.get_slot("symbol")
        print("symbol",symbol)
        url = "https://api.polygon.io/v3/reference/tickers/{}?apiKey={}".format(symbol,polygonapikey)
        r = json.loads(requests.get(url).content)["results"]
        market_cap = "Data Not Available"
        if "market_cap" in r.keys():
            market_cap = r["market_cap"]
        data = "{} \n The {} has a market cap of {}. \n The company has {} employees. \n The company went public on {} and has {} outstanding shares. \n For more information visit {}.".format(r["description"],symbol, market_cap, r["total_employees"], r["list_date"],r["share_class_shares_outstanding"], r["homepage_url"])
        dispatcher.utter_message(text=data)
        return []

class ActionGetTickerNews(Action):

    def name(self) -> Text:
        return "action_get_ticker_news"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print("---------action_get_ticker_news------------")
        symbol = tracker.get_slot("symbol")
        print("symbol", symbol)
        url = "https://api.polygon.io/v2/reference/news?apiKey={}&ticker={}&limit=3".format(polygonapikey, symbol)
        r = json.loads(requests.get(url).content)["results"]
        data = "Here's a list of trending news and articles on {} :-\n\n".format(symbol)
        count = 0
        if len(r) != 0:
            for i in r:
                count = count+1
                data = data+str(count)+"-"+i["title"]+"\n"+i["article_url"]+"\n\n"
        else:
            data = "The are no latest news or articles for {}".format(symbol)
        dispatcher.utter_message(text=data)
        return []

class ActionGetMarketHolidays(Action):

    def name(self) -> Text:
        return "action_get_market_holidays"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print("---------action_get_market_holidays------------")

        url = "https://api.polygon.io/v1/marketstatus/upcoming?apiKey={}".format(polygonapikey)
        r = json.loads(requests.get(url).content)[0]
        data = "The upcoming market holiday will fall on {} because of {}".format(r["date"],r["name"])
        dispatcher.utter_message(text=data)
        return []

class ActionGetMarketStatus(Action):

    def name(self) -> Text:
        return "action_get_market_status"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print("---------action_get_market_status------------")

        url = "https://api.polygon.io/v1/marketstatus/now?apiKey={}".format(polygonapikey)
        r = json.loads(requests.get(url).content)
        data = "The market is {} for today".format(r["market"])
        dispatcher.utter_message(text=data)
        return []

class ActionGetTickerDividends(Action):

    def name(self) -> Text:
        return "action_get_ticker_dividends"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print("---------action_get_ticker_dividend------------")
        symbol = tracker.get_slot("symbol")
        print("symbol", symbol)
        url = "https://api.polygon.io/v3/reference/dividends?ticker={}&limit=1&apiKey={}".format(symbol,polygonapikey)
        r = json.loads(requests.get(url).content)["results"]
        if len(r)!=0:
            r=r[0]
            data = "Below is the recent dividend information for {}:\n\nCash amount : {},{}\nDeclaration date : {}\nEx Dividend date : {}\nPay date : {}\nFrequency : {}".format(symbol,r["cash_amount"],r["currency"],r["declaration_date"],r["ex_dividend_date"],r["pay_date"],r["frequency"])
        else:
            data = "{} does not pay dividend".format(symbol)
        dispatcher.utter_message(text=data)
        return []

class ActionGetTickerSplits(Action):

    def name(self) -> Text:
        return "action_get_ticker_splits"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print("---------action_get_ticker_dividend------------")
        symbol = tracker.get_slot("symbol")
        print("symbol", symbol)
        url = "https://api.polygon.io/v3/reference/splits?ticker={}&apiKey={}".format(symbol,polygonapikey)
        r = json.loads(requests.get(url).content)["results"]
        if len(r)!=0:
            data = "A total of {} stock splits occured in {} history.\nThe most recent stock split occured on {} from {} share to {} shares.".format(len(r),symbol,r[0]["execution_date"],r[0]["split_from"],r[0]["split_to"])
        else:
            data = "No stock split has ever occured for {}".format(symbol)
        dispatcher.utter_message(text=data)
        return []

class ActionGetTickerDailyOpenClose(Action):

    def name(self) -> Text:
        return "action_get_ticker_daily_open_close"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val) >= 1:
            return "{:,.2f}".format(float(val))
        elif 1 > abs(val) >= 0.01:
            return "{:,.5f}".format(float(val))
        else:
            return "{:,.10f}".format(float(val))

    @staticmethod
    def prev_weekday(adate):
        adate -= timedelta(days=1)
        while adate.weekday() > 4:  # Mon-Fri are 0-4
            adate -= timedelta(days=1)
        return adate


    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print("---------action_get_ticker_daily_open_close------------")
        print(tracker.latest_message["entities"])
        try:
            time_stamp = next(tracker.latest_message["entities"][1]["value"],None)
        except:
            time_stamp = None

        if time_stamp == None:
            time_stamp = self.prev_weekday(date.today())
        else:
            time_stamp=time_stamp[:10]
        print("time_stamp",time_stamp)
        symbol = tracker.get_slot("symbol")


        print("symbol", symbol)
        url = "https://api.polygon.io/v1/open-close/{}/{}?adjusted=true&apiKey={}".format(symbol,time_stamp,polygonapikey)
        r = json.loads(requests.get(url).content)
        data = "{} price stats on {} :-\nOpen : ${}\nHigh : ${}\nLow : ${}\nClose : ${}\nVolume : {}\nAfter hours : ${}\nPre Market : ${}".format(symbol,time_stamp,self.roundval(r["open"]),self.roundval(r["high"]),self.roundval(r["low"]),self.roundval(r["close"]),self.roundval(r["volume"]),r["afterHours"],r["preMarket"])
        dispatcher.utter_message(text=data)
        return []

class ActionGetTickerGainersLosers(Action):

    def name(self) -> Text:
        return "action_get_ticker_gainers_losers"

    @staticmethod
    def roundval(val):
        if val == None:
            return val
        elif type(val) == str:
            return val
        elif abs(val) >= 1:
            return "{:,.2f}".format(float(val))
        elif 1 > abs(val) >= 0.01:
            return "{:,.5f}".format(float(val))
        else:
            return "{:,.10f}".format(float(val))

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print("---------action_get_ticker_gainers_losers------------")
        stocks_gainers = tracker.get_slot("stocks_gainers")
        url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers?apiKey={}".format(polygonapikey)
        r = json.loads(requests.get(url).content)["tickers"]
        data = "Below are the top {} for the day :-\n".format(stocks_gainers)
        for i in range(5):
            data= data + "{}- {} currently trading at {} with a percentage change of {} over the last trading day.\n".format(i+1,r[i]["ticker"],self.roundval(r[i]["todaysChangePerc"]),self.roundval(r[i]["lastQuote"]["P"]))

        dispatcher.utter_message(text=data)
        return []


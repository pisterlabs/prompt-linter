# Postprocessing for SBA DA scac:OKOSBA data extraction

import re
import time
from datetime import datetime
import openai
import json
from ._common import UnintendedFileException


def convert_dict_values_to_none(data: dict) -> dict:
    """fpo
    Recursively converts dictionary values to None.

    Args:
        data (dict): The dictionary to process.

    Returns:
        dict: The modified dictionary with values set to None.
    """
    if isinstance(data, dict):
        data_copy = data.copy()
        for key, value in data_copy.items():
            data_copy[key] = convert_dict_values_to_none(value)

    elif isinstance(data, list):
        data_copy = data.copy()
        tmp_list = []
        for item in data_copy:
            tmp_list.append(convert_dict_values_to_none(item))
        data_copy = tmp_list
    else:
        data_copy = None
    return data_copy


def call_llm(prompt, temperature=0.0):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-4-0613",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        timeout=1200
    )
    return completion.choices[0].message["content"]  # type: ignore


def parse_dict_by_rules(input_data, output_rules):
    prompt = f"""
        parse the following:
        {input_data}
        in the following format:
        {output_rules}
    """

    ret = call_llm(prompt)
    resp = json.loads(ret)
    return resp


def parse_ship_cons_data(text):
    prompt = f"""The following is the information of the shipper or consignee of a shipment.
            It contains the name of the shipper or consignee, in the following order: 'name', 'address', 'city', 'state', 'zip code', 'contact name', 'contact phone'.

            {text}

            """
    prompt += r"""
            From that, extract the following JSON:

            {
                "name": "string. Contains at start of the string before address dataExamples: '22775 LOWES OF SLIDELL 1684'. Default is ''.",
                "address": {
                    "street": "string. Only the street address. Do not put the full address here. Examples: '3301 VETERANS MEMORIAL BLVD SUITE #74', '123 Main St PO 
                    557', 'GES International Society for Labor 3761 Louisa St'. Do not add word 'Address'.",
                    "city": "string.",
                    "state": "string. The code of the state",
                    "postal_code": "string.",
                    "country_code": "string. Default is 'US'"
                },
                "contact": {
                    "name": "string. Examples: 'STORE PHONE # SEPHORA', 'RECEIVING', 'HALEY FEINGOLD', 'TIM MC TEE'. Default is ''.",
                    "tel": "string. Might not exist. Examples: '860-257-3300', '+15048304567'. Default is ''.",
                    "fax": "string. Might not exist. Default is ''."
                },
                "notes": "string. Might not exist. Default is ''."
            }

            Do not invent information. If the information is not there, leave it as null, unless a default is specified. Do not use examples as default values.
            """

    ret = call_llm(prompt)
    resp = json.loads(ret)
    return resp


def convert_time(string):
    try:
        time_obj = datetime.strptime(string, '%H:%M')
        res = {
            "time": time_obj.strftime('%H:%M'),
            "meridien": time_obj.strftime('%p'),
            "timezone": None
        }
    except:
        res = {
            "time": None,
            "meridien": None,
            "timezone": None
        }
    return res


class SBADA:
    def __init__(self, ret, output_format):
        self.ret = ret
        self.output_format = convert_dict_values_to_none(output_format)
        self.check_for_unsupported_file()

    def check_for_unsupported_file(self):
        if (
                self.ret.get("consignee") is None
                or self.ret.get("shipper") is None
                or self.ret.get("ot") is None
                or self.ret.get("dt") is None
                or self.ret.get("hawb") is None
        ):
            raise UnintendedFileException

    def get_goods_from_table(self, data):
        outer_data = []
        for row in data:
            clean_row_data = {}
            for k, v in row.value.items():
                clean_row_data.update({
                    k: v.value if v.value else v.content
                })
            if clean_row_data.get('Description'):
                row_data = {
                    "description": clean_row_data.get('Description'),
                    "dimensions": f"{clean_row_data.get('Length')} x {clean_row_data.get('Width')} x {clean_row_data.get('Height')}",
                    "weight": f"{clean_row_data.get('Weight')} {clean_row_data.get('Wgt UOM')}",
                    "pieces": clean_row_data.get('Containing Pcs')
                }
                outer_data.append(row_data)
        return outer_data

    async def run(self):
        # Output with default values
        self.output_format.update({
            'scac': 'OKOSBA',
            'shipment_identifier': self.ret.get("hawb", ""),
            'payment_method': 'PP',
            'handling': 'PUD',
            'BM': self.ret.get("hawb", ""),
            'MA': self.ret.get("mawb", ""),
            'SI': self.ret.get("si") if self.ret.get("si") else "",
            'OT': self.ret.get('ot'),
            'DT': self.ret.get('dt'),

            'shipment_date': self.ret.get("deliver_from_date", ""),
            'ready_date': self.ret.get("eta_date", ""),
            'delivery_date': self.ret.get("deliver_to_date", ""),

            'shipper': parse_ship_cons_data(self.ret.get('shipper')),
            'consignee': parse_ship_cons_data(self.ret.get('consignee')),
            "goods": {
                "pieces": self.get_goods_from_table(self.ret.get('goods_table')),
                "net_weight": self.ret.get('total_weight'),
                "no_of_pieces": self.ret.get('total_pieces'),
            },
            'ready_time': convert_time(self.ret.get('eta_time')),
            'shipment_time': convert_time(self.ret.get('deliver_from_time')),
            'delivery_time': convert_time(self.ret.get('deliver_to_time'))
        })

        return self.output_format

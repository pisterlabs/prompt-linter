# Postprocessing for iCat HAWB scac:ICAT data extraction

import re
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
            At first, exclude example: 'Booth: MEETING ROOM' from that. 
            Then, extract the following JSON:

            {
                "name": "string. Contains at start of the string before address data.
                 Examples: 'PEOPLE HELPING PEOPLE - TRAIN THE TRAINERS EVENT HYATT REGENCY DALLAS AMERICAN TRADESHOW SERVICES, LLC', 
                 'DISH NETWORK TEAM SUMMIT 2023 ROSEN SHINGLE CREEK AMERICAN TRADESHOW SERVICES, LLC', 
                 'UNCON/SWOOGO EVENT THE REDD AMERICAN TRADESHOW SERVICES, LLC', 
                 'PEOPLE HELPING PEOPLE - TRAIN THE TRAINERS EVENT 2 HYATT REGENCY DALLAS AMERICAN TRADESHOW SERVICES, LLC', 
                 'CONTACTPOINTE - HANWAH COURTYARD BY MARRIOTT NEW ORLEANS/GRETA CONTACTPOINTE'. Default is ''.",
                "address": {
                    "street": "string. Only the street address. Do not put the full address here. Examples: 
                    '3301 VETERANS MEMORIAL BLVD SUITE #74', 
                    '123 Main St PO 557', 
                    'GES International Society for Labor 3761 Louisa St', 
                    '5 WESTBANK EXPRESSWAY'. 
                    Do not add word 'Address'.",
                    "city": "string.",
                    "state": "string. The code of the state",
                    "postal_code": "string.",
                    "country_code": "string. Default is 'US'"
                },
                "contact": {
                    "name": "string. Examples: 'STORE PHONE # SEPHORA', 'RECEIVING', 'HALEY FEINGOLD', 'TIM MC TEE'. Default is ''.",
                    "tel": "string. Examples: '504-782-2268', '720-963-6300 X 311', '9858010674'. Convert to format XXX-XXX-XXXX. Default is ''.",
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
    if '-' in string:
        string = string.split('-')[-1]

    # clean string
    string = string.replace(' ', '')

    match = re.search(r'\d{1,2}:\d{2}\w{2}$', string)
    if match:
        string = match.group(0)

    time_obj = datetime.strptime(string, '%I:%M%p')
    res = {
        "time": time_obj.strftime('%H:%M'),
        "meridien": time_obj.strftime('%p'),
        "timezone": None
    }

    return res


class ICATHAWB:
    def __init__(self, ret, output_format):
        self.ret = ret
        self.output_format = convert_dict_values_to_none(output_format)
        self.check_for_unsupported_file()

    def check_for_unsupported_file(self):
        if (
                self.ret.get("consignee") is None
                or self.ret.get("shipper") is None
                or self.ret.get("OT") is None
                or self.ret.get("DT") is None
                or self.ret.get("HAWB") is None
        ):
            raise UnintendedFileException

    def get_goods_from_table(self, data):
        outer_data = []
        for row in data:
            clean_row_data = {}
            for k, v in row.value.items():
                if 'weight' in k.lower():
                    key_parts = k.split(' ')
                    tmp_k = key_parts[0]
                    tmp_val = v.value if v.value else v.content

                    clean_row_data.update({
                        tmp_k: tmp_val
                    })
                else:
                    clean_row_data.update({
                        k: v.value if v.value else v.content
                    })
            if clean_row_data.get('Description'):
                row_data = {
                    "description": clean_row_data.get('Description'),
                    "dimensions": f"{clean_row_data.get('Length')} X {clean_row_data.get('Width')} X {clean_row_data.get('Height')}",
                    "weight": f"{clean_row_data.get('Weight')}",
                    'pieces': int(float(clean_row_data.get('Pieces'))),
                    'package_type': clean_row_data.get('Pkg Type')
                }
                outer_data.append(row_data)
        return outer_data

    async def run(self):
        # Output with default values
        self.output_format.update({
            'scac': 'ICAT',
            'shipment_identifier': self.ret.get("HAWB", ""),
            'payment_method': 'PP',
            'handling': 'PUD',
            'BM': self.ret.get("HAWB", ""),
            'SI': self.ret.get("si") if self.ret.get("si") else "",
            'OT': self.ret.get('OT'),
            'DT': self.ret.get('DT'),

            'shipment_date': self.ret.get("date", ""),

            'delivery_date': self.ret.get("delivery_date", ""),

            'shipper': parse_ship_cons_data(self.ret.get('shipper')),
            'consignee': parse_ship_cons_data(self.ret.get('consignee')),
            "goods": {
                "pieces": self.get_goods_from_table(self.ret.get('goods_table')),
                "net_weight": self.ret.get('weight_'),
                "no_of_pieces": self.ret.get('pcs_'),
            },
            'ready_time': convert_time(self.ret.get('pickup_time')),
            'delivery_time': convert_time(self.ret.get('delivery_time'))
        })

        if self.ret.get("pickup_date"):
            pickup_date = datetime.strptime(self.ret.get("pickup_date"), '%B %d, %Y')
            self.output_format.update({
                'ready_date': pickup_date.strftime('%m/%d/%Y')
            })

        return self.output_format

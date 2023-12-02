# Postprocessing for Pegsus HAWB scac:PGAA data extraction

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
    prompt = f"""The following is the information of the shipper or consignee of a shipment. It contains the name of the shipper, its address and contact information.

            {text}

            """
    prompt += r"""
            From that, extract the following JSON:

            {
                "name": "string. Contains at start of the string before address dataExamples: '22775 LOWES OF SLIDELL 1684'",
                "address": {
                    "street": "string. Only the steet address. Do not put the full address here. Examples: '123 Main St', '123 Main St PO 557'",
                    "city": "string",
                    "state": "string. The code of the state",
                    "postal_code": "string",
                    "country_code": "string. Default is 'US'"
                },
                "contact": {
                    "name": "string. Examples: 'FOR AAMD 2023', 'HALEY FEINGOLD', 'SHIPPING', 'SHEPARD/TFORCE', 'TRE MOSELEY', 'DEB'. Default '' ",
                    "tel": "string. Might not exist. Examples: '860-257-3300'. Convert to format '000-000-0000'. Default ''",
                    "fax": "string. Might not exist. Default ''"
                },
                "notes": "string. Might not exist. Default ''"
            }

            Do not invent information. If the information is not there, leave it as null, unless a default is specified. Do not use examples as default values.
            """

    ret = call_llm(prompt)
    resp = json.loads(ret)
    return resp


class PegasusHAWB:
    def __init__(self, ret, output_format):
        self.ret = ret
        self.output_format = convert_dict_values_to_none(output_format)
        self.check_for_unsupported_file()

    def check_for_unsupported_file(self):
        if (
                self.ret.get("consignee") is None
                or self.ret.get("shipper") is None
                or self.ret.get("waybill") is None
                or self.ret.get("OT") is None
                or self.ret.get("DT") is None
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
                    if len(key_parts) > 1:
                        mesure = key_parts[1].replace('(', '').replace(')', '')
                        tmp_val = ' '.join([tmp_val, mesure])
                    clean_row_data.update({
                        tmp_k: tmp_val
                    })
                else:
                    clean_row_data.update({
                        k: v.value if v.value else v.content
                    })
            row_data = {
                "description": clean_row_data.get('Description'),
                "dimensions": f"{clean_row_data.get('Length')} x {clean_row_data.get('Width')} x {clean_row_data.get('Height')}",
                "weight": clean_row_data.get('Weight'),
                "pieces": clean_row_data.get('Pieces')
            }
            outer_data.append(row_data)
        return outer_data

    async def run(self):
        # Output with default values
        self.output_format.update({
            'scac': 'PGAA',
            'shipment_identifier': self.ret.get("waybill", ""),
            'payment_method': 'PP',
            'handling': 'PUD',
            'BM': self.ret.get("waybill", ""),
            'SI': self.ret.get("SI", ""),
            'OT': self.ret.get('OT'),
            'DT': self.ret.get('DT'),

            'shipment_date': self.ret.get("shipment_date", ""),
            'ready_date': self.ret.get('ready_date'),
            'delivery_date': self.ret.get('delivery_date'),

            'shipper': parse_ship_cons_data(self.ret.get('shipper')),
            'consignee': parse_ship_cons_data(self.ret.get('consignee')),
            "goods": {
                "pieces": self.get_goods_from_table(self.ret.get('goods_table')),
                "net_weight": self.ret.get('total_weight'),
                "no_of_pieces": self.ret.get('total_pieces'),
            },
        })

        # clean delivery date
        del_d_match = re.search('\d{1,2}/\d{2}/\d{2,4}', self.ret.get('delivery_date', ''))
        if del_d_match:
            self.output_format.update({
                'delivery_date': del_d_match.group(0)
            })

        if self.ret.get('ready_time'):
            time, meridien = self.ret.get('ready_time').split(' ')
            if '-' in time:
                time = time.split('-')[1]
            time_obj = datetime.strptime(f'{time} {meridien}', '%I:%M %p')
            self.output_format.update({
                'ready_time': {
                    "time": time_obj.strftime('%H:%M'),
                    "meridien": time_obj.strftime('%p'),
                    "timezone": None
                }
            })

        if self.ret.get('delivery_time'):
            match = re.search(r'\d{1,2}:\d{2} \w{2}', self.ret.get('delivery_time'))
            if match:
                delivery_time = match.group(0)

                time_obj = datetime.strptime(delivery_time, '%I:%M %p')
                self.output_format.update({
                    'delivery_time': {
                        "time": time_obj.strftime('%H:%M'),
                        "meridien": time_obj.strftime('%p'),
                        "timezone": None
                    }
                })

        return self.output_format

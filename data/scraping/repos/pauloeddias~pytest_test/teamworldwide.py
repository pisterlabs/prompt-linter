# Postprocessing for Team WorldWide scac:TAIF data extraction

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
    return completion.choices[0].message["content"] # type: ignore


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
                "name": "string. Examples: 'BEEKLEY MEDICAL C/O HERITAGE TFORCE FRT C/O EXHIBIT TRANSFR', 'TSA', 'TPS DISPLAYS'",
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


def clean_dimensions(text):
    res = ''
    match = re.findall(r'\d{1,2}@\d{1,3}X\d{1,3}X\d{1,3}', text)
    if match:
        res = ' '.join(match)
    return res


def extract_delivery_date_and_time(text):
    res = {}
    date_time_regex = r'(?P<delivery_date>\d{8})( (AT|BY) (?P<delivery_time>\d{4}))?'
    match = re.search(date_time_regex, text)
    if match:
        res = match.groupdict()
    return res


class TeamWWPostprocessing:
    def __init__(self, ret, output_format):
        self.ret = ret
        self.output_format = convert_dict_values_to_none(output_format)
        self.check_for_unsupported_file()

    def check_for_unsupported_file(self):
        if (
            self.ret.get("shipper") is None
            or self.ret.get("consignee") is None
            or self.ret.get("ship_id") is None
            or self.ret.get("DT") is None
            or self.ret.get("OT") is None
        ):
            raise UnintendedFileException

    async def run(self):

        self.parse_output_format()

        return self.output_format

    def parse_output_format(self):
        # Output with default values
        dimensions_list = clean_dimensions(self.ret.get("dimension_data", "")).split()
        pieces = []
        for dim in dimensions_list:
            dim_with_pcs = dim.split('@')
            piece_item = {
                'description': self.ret.get("piece_description", ""),
                'dimensions': dim_with_pcs[1].lower().replace('x', ' X ') if len(dim_with_pcs) > 1 else None,
                'weight':  None,
                'pieces': dim_with_pcs[0] if len(dim_with_pcs) > 1 else None,
            }
            pieces.append(piece_item)

        self.output_format.update({
            'scac': 'TAIF',
            'shipment_identifier': self.ret.get("ship_id", ""),
            'payment_method': 'PP',
            'handling': 'PUD',
            'PO': self.ret.get('po_number'),
            'MA': self.ret.get("ship_id", ""),
            'BM': self.ret.get("airbill", ""),
            'SI': self.ret.get("SI", ""),
            'OT': self.ret.get("OT", ""),
            'DT': self.ret.get("DT", ""),
            'carrier': {
                'name': self.ret.get("carier", ""),
                "address": {
                    "street": None,
                    "city": None,
                    "state": None,
                    "postal_code": None,
                    "country_code": None,
                },
                "contact": {"name": None, "tel": None, "fax": None},
            },
            'goods': {
                'net_weight': self.ret.get("weight", ""),
                'no_of_pieces': self.ret.get("pcs", ""),
                'pieces': pieces
            },

            'shipment_date': self.ret.get("ship_date", ""),
            'shipment_time': {
                'time': self.ret.get("ship_time", ""),
                'timezone': '',
                'meridien': ''
            },
            'delivery_date': self.ret.get("deliver_date", ""),
            'delivery_time': {
                'time': self.ret.get("deliver_time", ""),
                'timezone': '',
                'meridien': ''
            },
            'ready_date': self.ret.get("ETA", ""),
            'ready_time': {
                'time': self.ret.get('ship_time', ''),
                'timezone': '',
                'meridien': ''
            },
            'shipper': parse_ship_cons_data(self.ret.get('shipper')),
            'consignee': parse_ship_cons_data(self.ret.get('consignee')),
        })

        self.ret.update({**extract_delivery_date_and_time(self.ret.get('SI'))})

        # Need to clean output data with openai
        input_data = {
            'shipment_date': self.output_format.get('shipment_date', ''),
            'shipment_time': self.output_format.get('shipment_time', {}).get('time', ''),
            'ready_date': self.output_format.get('ready_date', ''),
            'ready_time': self.ret.get('ship_time', ''),
            'delivery_date': self.ret.get('delivery_date', ''),
            'delivery_time': self.ret.get('delivery_time', ''),
        }

        output_rules = """{
            "shipment_date": string, // 
                Extract from shipment_date. Return in format MM/DD/YYYY.
            "shipment_time": string, // 
                Extract from shipment_time. Return in format HH:MM AM or PM. Example: 16:00 PM

            "ready_date": string, // 
                Extract from ready_date. Return in format MM/DD/YYYY.
            "ready_time": string, // 
                Extract from ready_time. Return in format HH:MM AM or PM. Example: 16:00 PM

            "delivery_date": string, // 
                Extract from delivery_date. Return in format MM/DD/YYYY.
            "delivery_time": string, // 
                Extract from delivery_time. Return in format HH:MM AM or PM. Example: 16:00 PM

        }"""

        cleaned_data = parse_dict_by_rules(input_data, output_rules)

        for k, v in cleaned_data.items():
            if k in ['delivery_time', 'ready_time', 'shipment_time']:
                if v:
                    v_time = datetime.strptime(v, '%I:%M %p')
                    self.output_format[k].update({
                        'time': v_time.strftime('%H:%M'),
                        'meridien': v_time.strftime('%p').upper()
                    })
            else:
                self.output_format.update({k: v})

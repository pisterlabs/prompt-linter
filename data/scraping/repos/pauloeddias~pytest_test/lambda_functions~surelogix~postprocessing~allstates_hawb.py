import asyncio
import openai
import json

from .BasePostprocessor import BasePostprocessor


class AllStatesHAWBPostprocessing(BasePostprocessor):
    def __init__(self, ret, output_format):
        self.ret = ret
        self.output_format = output_format
        self.check_for_unsupported_file()

    def check_for_unsupported_file(self):
        if (
            self.ret.get("shipper_name") == None
            or self.ret.get("consignee_name") == None
            or self.ret.get("DT") == None
            or self.ret.get("hwb") == None
            or self.ret.get("OT") == None
            or self.ret.get("scac") == None
        ):
            raise Exception("Unintended file")


    def parse_output_format(self, shipper, consignee, delivery_date, delivery_time, shipment_date, shipment_time):
        for i in self.output_format:
            self.output_format[i] = None

        self.output_format['shipper'] = shipper
        self.output_format['consignee'] = consignee
        self.output_format['goods'] = {
            "net_weight": self.ret.get('wt'),
            "no_of_pieces": self.ret.get('pcs'),
            'pieces': self.ret.get('goods_table')
        }
        self.output_format['shipment_identifier'] = self.ret.get('hwb')
        self.output_format['BM'] = self.ret.get('hwb')
        self.output_format['MA'] = self.ret.get('mawb')
        self.output_format['scac'] = self.ret.get('scac')
        self.output_format['shipment_date'] = shipment_date
        self.output_format['shipment_time'] = shipment_time
        self.output_format['payment_method'] = 'PP'
        self.output_format['PO'] = None
        self.output_format['SI'] = self.ret.get('SI')
        self.output_format['OT'] = self.ret.get('OT')
        self.output_format['DT'] = self.ret.get('DT')
        self.output_format['handling'] = 'PUC'
        self.output_format['delivery_date'] = delivery_date
        self.output_format['delivery_time'] = delivery_time
        self.output_format['BM'] = self.ret.get('hwb')
        self.output_format['PRG'] = None
        self.output_format['REF'] = None
        self.output_format['CR'] = None
        self.output_format['BOL'] = None




    async def run(self):
        shipper_prompt = self.get_shipper_prompt()

        consignee_prompt = self.get_consignee_prompt()

        delivery_date_prompt = self.get_delivery_date_and_time_prompt(self.ret["delivery_date"], self.ret["delivery_time"])

        shipment_date_prompt = self.get_shipment_date_and_time_prompt(self.ret["shipment_date_time"])


        resps = await self.call_llm_multiple_prompts_async(
            [shipper_prompt, consignee_prompt, delivery_date_prompt, shipment_date_prompt]
        )

        self.get_goods_table()

        shipper = json.loads(resps[0])
        consignee = json.loads(resps[1])
        delivery_date = json.loads(resps[2])
        shipment_date = json.loads(resps[3])

        self.parse_output_format(shipper, consignee,delivery_date.get('delivery_date'),delivery_date.get('delivery_time'),shipment_date.get('shipment_date'),shipment_date.get('shipment_time'))

        return self.output_format

    def get_shipper_prompt(self):

        shipper = self.ret['shipper_name']

        if self.ret.get('shipper_address_1'):
            shipper += '\n' +self.ret.get('shipper_address_1')

        if self.ret.get('shipper_address_2'):
            shipper += '\n' + self.ret.get('shipper_address_2')

        if self.ret.get('shipper_city'):
            shipper += '\n' + self.ret.get('shipper_city')

        if self.ret.get('shipper_phone'):
            shipper += '\n' + self.ret.get('shipper_phone')


        prompt = f"""The following is the information of the shipper of a shipment. It contains the name of the shipper, its address and contact information.

        {shipper}

        """
        prompt += r"""
        From that, extract the following JSON:

        {
            "name": "string",
            "address": {
                "street": "string. Only the street address. May or may not contain a PO Number. Do not put the full address here. Examples: '123 Main St', '123 Main St PO 557'",
                "city": "string",
                "state": "string. The code of the state",
                "postal_code": "string",
                "country_code": "string. Default is 'US'"
            },
            "contact": {
                "name": "string",
                "tel": "string",
                "fax": "string. Might not exist"
            }
        }

        Do not invent information. If the information is not there, leave it as null, unless a default is specified.
        """

        return prompt

    def get_consignee_prompt(self):

        consignee = self.ret['consignee_name']

        if self.ret.get('consignee_address_1'):
            consignee += '\n' + self.ret.get('consignee_address_1')

        if self.ret.get('consignee_address_2'):
            consignee += '\n' + self.ret.get('consignee_address_2')

        if self.ret.get('consignee_city'):
            consignee += '\n' + self.ret.get('consignee_city')

        if self.ret.get('consignee_phone'):
            consignee += '\n' + self.ret.get('consignee_phone')


        prompt = f"""The following is the information of the consignee of a shipment. It contains the name of the consignee, its address and contact information.

        {consignee}

        """
        prompt += r"""
        From that, extract the following JSON:

        {
            "name": "string",
            "address": {
                "street": "string. Only the steet address. May or may not contain a PO Number. Do not put the full address here. Examples: '123 Main St', '123 Main St PO 557'",
                "city": "string",
                "state": "string. The code of the state",
                "postal_code": "string",
                "country_code": "string. Default is 'US'"
            },
            "contact": {
                "name": "string",
                "tel": "string",
                "fax": "string. Might not exist"
            }
        }

        Do not invent information. If the information is not there, leave it as null, unless a default is specified.
        """
        return prompt

    def get_shipment_date_and_time_prompt(self, shipment_date_time):
        prompt = f"""The following is the information of the shipment date and time of a shipment. It contains the shipment date and time.

                {shipment_date_time}
                
                """
        prompt += r"""
                From that, extract the following JSON:

                {
                    "shipment_date": "string. The date of delivery. Format: MM/DD/YYYY",
                    "shipment_time": {
                            "time": "string. The time of delivery. Format: HH:MM",
                            "meridien": "If it's AM or PM",
                            "timezone": "New Orleans time zone for that date. Format: 'UTC-5'",
                        }
                    
                }

                Do not invent information. If the information is not there, leave it as null.
                """

        return prompt

    def get_delivery_date_and_time_prompt(self, delivery_date, delivery_time):
        prompt = f"""The following is the information of the delivery date and time of a shipment. It contains the delivery date and possibly the delivery time.

                {delivery_date} {delivery_time}

                """
        prompt += r"""
                From that, extract the following JSON:

                {
                    "delivery_date": "string. The date of delivery. Format: MM/DD/YYYY",
                    "delivery_time": {
                            "time": "string. The time of delivery. Format: HH:MM",
                            "meridien": "If it's AM or PM",
                            "timezone": "New Orleans time zone for that date. Format: 'UTC-5'",
                        }

                }

                Do not invent information. If the information is not there, leave it as null.
                """

        return prompt

    def get_goods_table(self):
        goods = []
        for x in self.ret['goods_table']:
            pcs = None
            de = None
            wi = None
            he = None
            le = None
            we = None
            ty = None
            dims = None
            if 'pcs' in x.value:
                pcs = int(x.value['pcs'].value)
            if 'description' in x.value:
                de = x.value['description'].value
            if 'width' in x.value:
                wi = float(x.value['width'].value)
            if 'height' in x.value:
                he = float(x.value['height'].value)
            if 'length' in x.value:
                le = float(x.value['length'].value)
            if 'weight' in x.value:
                we = float(x.value['weight'].value)
            if 'Type' in x.value:
                ty = x.value['Type'].value
            if pcs and le and wi and he:
                dims = f'{le} X {wi} X {he}'
            goods.append({
                'description': de,
                'pieces': pcs,
                'dimensions': dims,
                'weight': we,
                'package_type': ty,
            })
        self.ret['goods_table'] = goods

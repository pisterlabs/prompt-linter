import asyncio
import openai
import json

from .BasePostprocessor import BasePostprocessor
from ._common import UnintendedFileException


class PegasusDAPostprocessor(BasePostprocessor):
    def __init__(self, ret, output_format):
        self.ret = ret
        self.output_format = output_format
        self.check_for_unsupported_file()

    def check_for_unsupported_file(self):

        if (
            self.ret.get("shipper") == None
            or self.ret.get("consignee") == None
            or self.ret.get("ship_date_time") == None
            or self.ret.get("scac") == None
        ):
            raise UnintendedFileException


    def parse_output_format(self,**kwargs):
        self.output_format = self.convert_dict_values_to_none(self.output_format)

        self.output_format['shipper'] = kwargs.get('shipper')
        self.output_format['consignee'] = kwargs.get('consignee')
        self.output_format['goods'] = {
            "net_weight": float(self.ret.get('weight')),
            "no_of_pieces": int(self.ret.get('pcs')),
            'pieces': [{
                "description": None,
                "dimensions": None,
                "weight": None,
                'pieces': None,
                'package_type': None
            }]
        }
        self.output_format['shipment_identifier'] = self.ret.get('hwb')
        self.output_format['MA'] = self.ret.get('mawb')
        self.output_format['scac'] = self.ret.get('scac')
        self.output_format['shipment_date'] = kwargs.get('shipment_date')
        self.output_format['shipment_time'] = kwargs.get('shipment_time')
        self.output_format['payment_method'] = 'PP'
        self.output_format['SI'] = self.ret.get('SI')
        if self.output_format['SI']:
            self.output_format['SI'] = self.output_format['SI'].replace('*','')
        self.output_format['handling'] = 'Delivery Alert'
        self.output_format['delivery_date'] = kwargs.get('delivery_date')
        self.output_format['delivery_time'] = kwargs.get('delivery_time')
        self.output_format['BM'] = self.ret.get('hwb')

        self.output_format['carrier']['name'] = self.ret.get('carrier')
        self.output_format['OT'] = self.ret.get('OT')
        self.output_format['DT'] = self.ret.get('DT')




    async def run(self):
        shipper_prompt = self.get_shipper_prompt(self.ret.get('shipper'))

        consignee_prompt = self.get_consignee_prompt(self.ret.get('consignee'))

        delivery_date_time = 'Date:'+self.ret.get('delivery_date', '') + '\n' + 'Time:'+self.ret.get('delivery_time', '')
        shipment_date_time = self.ret.get('ship_date_time', '')

        delivery_date_prompt = self.get_delivery_date_and_time_prompt(delivery_date_time, shipment_date_time)
        shipment_date_prompt = self.get_shipment_date_and_time_prompt(shipment_date_time)

        resps = await self.call_llm_multiple_prompts_async(
            [shipper_prompt, consignee_prompt,delivery_date_prompt,shipment_date_prompt]
        )

        shipper = json.loads(resps[0])
        consignee = json.loads(resps[1])

        delivery = json.loads(resps[2])
        delivery_date = delivery.get('delivery_date', '')
        delivery_time = delivery.get('delivery_time', '')

        shipment = json.loads(resps[3])
        shipment_date = shipment.get('shipment_date', '')
        shipment_time = shipment.get('shipment_time', '')

        self.parse_output_format(shipper=shipper,
                                 consignee=consignee,
                                 delivery_date=delivery_date,
                                 delivery_time=delivery_time,
                                 shipment_date=shipment_date,
                                 shipment_time=shipment_time)

        return self.output_format

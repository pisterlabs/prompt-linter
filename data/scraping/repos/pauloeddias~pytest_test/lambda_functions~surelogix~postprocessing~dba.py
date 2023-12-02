import asyncio
import openai
import json
from ._common import UnintendedFileException
from .BasePostprocessor import BasePostprocessor
from ._common import UnintendedFileException


async def call_llm_async(prompt, temperature=0.0):
    completion = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        # model="gpt-4-0613",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return completion.choices[0].message["content"]


async def call_llm_multiple_prompts_async(prompts):
    tasks = []
    for prompt in prompts:
        tasks.append(asyncio.create_task(call_llm_async(prompt)))
    resp = await asyncio.gather(*tasks)
    return resp


class DBADeliveryAlertPostprocessing:

    def __init__(self, ret, output_format):
        self.ret = ret
        self.output_format = output_format

        self.check_for_unsupported_file()

    def check_for_unsupported_file(self):
        if (
                not self.ret.get('scac') or
                'distribution by air' not in self.ret.get('scac').lower().strip() or
                not self.ret.get('file_type') or
                self.ret.get('file_type', '').lower().strip() != 'delivery alert'
        ):
            raise UnintendedFileException

    def parse_output_format(self, **kwargs):
        for i in self.output_format:
            self.output_format[i] = None

        self.output_format["scac"] = "DBYA"
        self.output_format["handling"] = "PUD"  # <-- in the mapping just 'hard codded'
        self.output_format["payment_method"] = "PP"

        self.output_format["OT"] = self.ret.get("OT")
        self.output_format["DT"] = self.ret.get("DT")

        self.output_format["shipment_identifier"] = self.ret.get("HAWB")
        self.output_format["BM"] = self.ret.get("HAWB")
        self.output_format["MA"] = self.ret.get("HAWB")

        self.output_format["SI"] = self.ret.get("SI")

        self.output_format["delivery_date"] = self.ret.get("delivery_date")
        self.output_format["delivery_time"] = {
            "time":
                self.ret.get("delivery_time").replace('to', '-') if self.ret.get("delivery_time") else self.ret.get(
                    "delivery_time"),
            "meridien": None,
            "timezone": None,
        }

        self.output_format["ready_date"] = self.ret.get("ETA_date")
        self.output_format["ready_time"] = {
            "time": self.ret.get("ETA_time"),
            "meridien": None,
            "timezone": None,
        }

        self.output_format["shipper"] = {
            "name": self.ret.get("shipper"),
            "address": {
                "street": None,
                "city": None,
                "state": None,
                "postal_code": None,
                "country_code": None
            },
            "contact": {
                "name": None,
                "tel": None,
                "fax": None
            },
            "notes": None
        }

        self.output_format["consignee"] = kwargs.get('consignee')

        self.output_format["goods"] = {
            "net_weight": self.ret.get('weight_'),
            "no_of_pieces": self.ret.get('pcs_'),
            "pieces": self.get_pieces_from_table(self.ret['goods_table']),
        }

    async def run(self):

        consignee_prompt = self.get_consignee_prompt(self.ret["consignee"])

        resps = await call_llm_multiple_prompts_async(
            [consignee_prompt]
        )

        consignee = json.loads(resps[0])

        self.parse_output_format(
            consignee=consignee
        )

        return self.output_format

    def get_consignee_prompt(self, consignee):
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
                "name": "string Example: PAUL Anderson",
                "tel": "string Example: 111-111-1111",
                "fax": "string. Might not exist"
            }
        }

        Do not invent information. If the information is not there, leave it as null, unless a default is specified.
        """
        return prompt

    def get_pieces_from_table(self, goods_table):
        pieces = []

        if goods_table:
            for i in range(len(goods_table)):
                dict_row = self.parse_1d_table_into_dict(goods_table[i])

                pieces_item = {
                    'description': dict_row.get('Marks & Numbers'),
                    'dimensions': ' x '.join([dict_row.get('Length'), dict_row.get('Width'), dict_row.get('Height')]),
                    'weight': dict_row.get('Weight'),
                    'package_type': dict_row.get('Type'),
                    'pieces': dict_row.get('Pcs'),
                }

                pieces.append(pieces_item)

        return pieces

    def parse_1d_table_into_dict(self, table):
        fields_dict = {}
        try:
            for key, field in table.value.items():
                fields_dict[key] = field.value

        except Exception as e:
            print(f" ~~~ ERROR PARSE TABLE INTO DICT ~~~ '{e}!")
        return fields_dict

class DBAPickupAlertPostprocessing(BasePostprocessor):
    def __init__(self, ret, output_format):
        self.ret = ret
        self.output_format = output_format
        self.check_for_unsupported_file()

    def check_for_unsupported_file(self):
        if (
                self.ret.get("shipper_line1") == None
                or self.ret.get("consignee_line1") == None
                or self.ret.get("hwb") == None
                or self.ret.get("OT") == None
                or self.ret.get("scac") == None
        ):
            raise UnintendedFileException

    def parse_output_format(self, shipper, consignee, shipment_date, shipment_time):
        for i in self.output_format:
            self.output_format[i] = None

        self.output_format['shipper'] = shipper
        self.output_format['consignee'] = consignee

        total_weight = self.ret.get('wt')
        if not total_weight:
            if 'goods_table' in self.ret:
                #sum the values in the weight column
                try:
                    total_weight = sum([float(x['weight']) for x in self.ret['goods_table']])
                except:
                    total_weight = 0

        total_pieces = self.ret.get('pcs')
        if not total_pieces:
            if 'goods_table' in self.ret:
                #sum the values in the pieces column
                try:
                    total_pieces = sum([int(x['pieces']) for x in self.ret['goods_table']])
                except:
                    total_pieces = 0
        else:
            try:
                total_pieces = int(total_pieces)
            except:
                pass



        self.output_format['goods'] = {
            "net_weight": total_weight,
            "no_of_pieces": total_pieces,
            'pieces': self.ret.get('goods_table')
        }
        self.output_format['shipment_identifier'] = self.ret.get('hwb')
        self.output_format['BM'] = self.ret.get('hwb')
        self.output_format['scac'] = 'DBYA'
        self.output_format['shipment_date'] = shipment_date
        self.output_format['shipment_time'] = shipment_time
        self.output_format['payment_method'] = 'PP'
        self.output_format['PO'] = None
        self.output_format['SI'] = self.ret.get('SI')
        self.output_format['pickup_instructions'] = self.ret.get('pickup_instructions')
        self.output_format['OT'] = self.ret.get('OT')
        self.output_format['handling'] = 'PUC'
        self.output_format['service_code'] = 'PUC'
        self.output_format['BM'] = self.ret.get('hwb')

    async def run(self):
        shipper_prompt = f"""
        shipper name: {self.ret.get('shipper_line1')}
        {self.ret.get('shipper_line2')}
        {self.ret.get('shipper_line3')}
        {self.ret.get('shipper_line4')}
        contact name: {self.ret.get('shipper_contact')}
        phone: {self.ret.get('shipper_phone')}
        """
        shipper_prompt = self.get_shipper_prompt(shipper_prompt)

        consignee_prompt = f"""
                consignee name: {self.ret.get('consignee_line1')}
                {self.ret.get('consignee_line2')}
                {self.ret.get('consignee_line3')}
                {self.ret.get('consignee_line4')}
                contact name: {self.ret.get('consignee_contact')}
                phone: {self.ret.get('consignee_phone')}
                """
        consignee_prompt = self.get_consignee_prompt(consignee_prompt)


        shipment_date_prompt = self.get_shipment_date_and_time_prompt(self.ret["pickup_date"]+'  '+self.ret["pickup_time"])

        resps = await self.call_llm_multiple_prompts_async(
            [shipper_prompt, consignee_prompt, shipment_date_prompt]
        )

        self.get_goods_table()

        shipper = json.loads(resps[0])
        consignee = json.loads(resps[1])
        shipment_date = json.loads(resps[2])

        self.parse_output_format(shipper, consignee, shipment_date.get('shipment_date'),
                                 shipment_date.get('shipment_time'))

        return self.output_format

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


    def get_goods_table(self):
        goods = []
        for x in self.ret['goods_table']:
            try:
                pcs = None
                de = None
                wi = None
                he = None
                le = None
                we = None
                ty = None
                if 'pieces' in x.value:
                    pcs = int(x.value['pieces'].value)
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

                dims=''
                if pcs:
                    dims = f'{pcs} @ '
                else:
                    dims = '0 @ '

                if le:
                    dims = dims + f'{le} x '
                else:
                    dims = dims + '0 x '

                if wi:
                    dims = dims + f'{wi} x '
                else:
                    dims = dims + '0 x '

                if he:
                    dims = dims + f'{he}'
                else:
                    dims = dims + '0'
                dims = dims.split('@')[1].strip().replace('×', 'x')
                goods.append({'description': de, 'dimensions': dims, 'weight': we, 'pieces': pcs})
            except:
                pass
        self.ret['goods_table'] = goods

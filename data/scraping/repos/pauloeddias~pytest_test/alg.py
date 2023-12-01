import asyncio
import openai
import json

from .BasePostprocessor import BasePostprocessor


class ALGPostprocessing(BasePostprocessor):
    def __init__(self, ret, output_format):
        self.ret = ret
        self.output_format = output_format
        self.check_for_unsupported_file()

    def check_for_unsupported_file(self):
        if (
            self.ret.get("shipper") == None
            or self.ret.get("consignee") == None
            or self.ret.get("DT") == None
            or self.ret.get("HWB") == None
            or self.ret.get("OT") == None
            or self.ret.get("MAWB") == None
            or self.ret.get("scac") == None
        ):
            raise Exception("Unintended file")


    def parse_output_format(self, shipper, consignee, delivery_date, delivery_time):
        for i in self.output_format:
            self.output_format[i] = None

        self.output_format
        self.output_format["scac"] = "AAIE"
        self.output_format["OT"] = self.ret.get("OT")
        self.output_format["DT"] = self.ret.get("DT")
        self.output_format["shipment_identifier"] = self.ret.get("HWB")
        self.output_format["BM"] = self.ret.get("HWB")
        self.output_format["shipment_date"] = self.ret.get("ship_date")
        self.output_format["shipment_time"] = {
            "time": None,
            "meridien": None,
            "timezone": None,
        }
        self.output_format["SI"] = self.ret.get("SI")

        self.output_format["MA"] = self.ret.get("MAWB")
        self.output_format["handling"] = "PUD"

        self.output_format["carrier"] = {
            "name": self.ret.get("carrier"),
            "address": {
                "street": None,
                "city": None,
                "state": None,
                "postal_code": None,
                "country_code": None,
            },
            "contact": {"name": None, "tel": None, "fax": None},
        }

        pieces = []
        for dim in self.ret.get("dims_table"):
            pieces.append(dim)

        self.output_format["goods"] = {
            "net_weight": str(self.ret.get("weight")),
            "no_of_pieces": self.ret.get("pcs"),
            "pieces": pieces,
        }

        self.output_format['ready_date'] = self.ret.get('EDA')
        self.output_format["ready_time"] = {
            "time": self.ret.get('ETA'),
            "meridien": None,
            "timezone": None,
        }
        self.output_format['PO'] = self.ret.get('PO')

        self.output_format["shipper"] = shipper
        self.output_format["consignee"] = consignee
        self.output_format["delivery_date"] = delivery_date
        if delivery_time is not None:
            self.output_format["delivery_time"] = delivery_time
        else:
            self.output_format["delivery_time"] = {
                "time": None,
                "meridien": None,
                "timezone": None,
            }

    def parse_dims_table(self, dims_table):
        if dims_table is not None:
            dims = []
            for x in dims_table:
                pcs = x.value['Pieces'].value
                le = x.value['Length'].value
                wi = x.value['Width'].value
                he = x.value['Height'].value

                dims.append({
                    'pieces': pcs,
                    'dimensions': f"{le} X {wi} X {he}",
                    'description': self.ret.get("pcs_description"),
                    'weight': None
                })

            return dims
        else:
            return []

    async def run(self):
        shipper_prompt = self.get_shipper_prompt(self.ret["shipper"])

        consignee_prompt = self.get_consignee_prompt(self.ret["consignee"])

        delivery_date_prompt = self.get_delivery_date_and_time_prompt(self.ret["deliver_date"], self.ret["ship_date"])

        if 'dims_table' in self.ret:
            dims_table = self.parse_dims_table(self.ret['dims_table'])
            self.ret['dims_table'] = dims_table

        resps = await self.call_llm_multiple_prompts_async(
            [shipper_prompt, consignee_prompt, delivery_date_prompt]
        )

        shipper = json.loads(resps[0])
        consignee = json.loads(resps[1])
        delivery_date = json.loads(resps[2])

        self.parse_output_format(shipper, consignee,delivery_date.get('delivery_date'),delivery_date.get('delivery_time'))

        return self.output_format


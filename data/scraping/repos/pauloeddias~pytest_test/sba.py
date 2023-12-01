import asyncio
import openai
import json
from .BasePostprocessor import BasePostprocessor
import re
from ._common import UnintendedFileException


class SBAPostprocessing(BasePostprocessor):
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
            or 'sba' not in self.ret.get('scac').lower()
        ):
            raise UnintendedFileException



    def parse_output_format(self, shipper, consignee, destination):
        for i in self.output_format:
            self.output_format[i] = None
        self.output_format
        self.output_format["scac"] = "OKOSBA"
        self.output_format["OT"] = self.ret.get("OT")
        self.output_format["shipment_identifier"] = self.ret.get("HWB")
        self.output_format["BM"] = self.ret.get("HWB")
        self.output_format["shipment_date"] = self.ret.get("ship_date")
        self.output_format["shipment_time"] = {
            "time": self.ret.get("ship_time"),
            "meridien": None,
            "timezone": None,
        }
        self.output_format["SI"] = self.ret.get("SI")
        self.output_format["delivery_date"] = self.ret.get("deliver_date")
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
        #print('RET<<<', self.ret)

        print('<<self.ret.get("dims_table")<<', self.ret.get("dims_table"))
        self.output_format["goods"] = {
            "net_weight": str(self.ret.get("weight")),
            "no_of_pieces": self.ret.get("pcs"),
            "pieces": self.get_pieces(),
        }

        self.output_format["ready_time"] = {
            "time": None,
            "meridien": None,
            "timezone": None,
        }
        self.output_format["delivery_time"] = {
            "time": None,
            "meridien": None,
            "timezone": None,
        }

        self.output_format["shipper"] = shipper
        self.output_format["consignee"] = consignee
        self.output_format["DT"] = destination

    def get_pieces(self):
        pieces = []

        pattern = r"\d+ @ \d+ x \d+ x \d+"
        matches = re.findall(pattern, str(self.ret.get('dims_table')))

        for match in matches:
            pc = match.split('@')[0].strip()
            dimensions = match.split('@')[1].strip()
            piece = {
                "description": self.ret.get("pcs_description"),
                "dimensions": dimensions.lower().replace('x', 'X'),
                "pieces": pc,
                "weight": None,
            }
            pieces.append(piece)

        return pieces

    async def run(self):
        shipper_prompt = self.get_shipper_prompt(self.ret["shipper"])

        consignee_prompt = self.get_consignee_prompt(self.ret["consignee"])

        destination_prompt = self.get_destination_prompt(self.ret.get("DT"))

        resps = await self.call_llm_multiple_prompts_async(
            [shipper_prompt, consignee_prompt,destination_prompt]
        )

        shipper = json.loads(resps[0])
        consignee = json.loads(resps[1])
        destination = resps[2]

        self.parse_output_format(shipper, consignee, destination)

        return self.output_format



    def get_destination_prompt(self,destination):

        prompt =f"""
        I want to extract the airport code from the following:
        
        {destination}
        
        Only respond with the airport code found.
        """

        return prompt


class SBADRPostprocessing(SBAPostprocessing):
    """ SBA Delivery Receipt """

    def check_for_unsupported_file(self):
        if (
                not self.ret.get('scac') or
                self.ret.get('scac', '').lower().strip() != 'sba' or
                not self.ret.get('file_type') or
                self.ret.get('file_type', '').lower().strip() != 'received freight in good order'
                # file_type here is a workaround, since the documents don't have 'delivery receipt' in them
        ):

            raise UnintendedFileException

    def parse_output_format(self, *args, **kwargs):

        for i in self.output_format:
            self.output_format[i] = None

        self.output_format["scac"] = "OKOSBA"
        self.output_format["OT"] = self.ret.get("OT")
        self.output_format["DT"] = self.ret.get("DT")

        self.output_format["shipment_identifier"] = self.ret.get("hwab")
        self.output_format["BM"] = self.ret.get("hwab")
        self.output_format["MA"] = self.ret.get("hwab")

        self.output_format["shipment_date"] = self.ret.get("shipment_date")
        self.output_format["shipment_time"] = {
            "time": None,
            "meridien": None,
            "timezone": None,
        }
        self.output_format["SI"] = self.ret.get("SI")
        self.output_format["delivery_date"] = self.ret.get("delivery_date")
        if self.output_format["delivery_date"]:
            self.output_format["delivery_date"] = self.output_format["delivery_date"].replace('by:', '').strip()

        self.output_format["delivery_time"] = {
            "time": None,
            "meridien": None,
            "timezone": None,
        }
        self.output_format["handling"] = "PUD"

        self.output_format["carrier"] = {
            "name": None,
            "address": {
                "street": None,
                "city": None,
                "state": None,
                "postal_code": None,
                "country_code": None,
            },
            "contact": {"name": None, "tel": None, "fax": None},
        }

        self.output_format["goods"] = {
            "net_weight": self.ret.get("weight_"),
            "no_of_pieces": self.ret.get("pcs_"),
            "pieces": self.get_pieces(),
        }

        self.output_format["ready_time"] = {
            "time": None,
            "meridien": None,
            "timezone": None,
        }

        self.output_format["shipper"] = kwargs.get('shipper')
        self.output_format["consignee"] = kwargs.get('consignee')

    async def run(self):
        shipper_prompt = self.get_shipper_prompt(self.ret["shipper"])

        consignee_prompt = self.get_consignee_prompt(self.ret["consignee"])

        resps = await self.call_llm_multiple_prompts_async(
            [shipper_prompt, consignee_prompt]
        )

        shipper = json.loads(resps[0])
        consignee = json.loads(resps[1])

        self.parse_output_format(
            shipper=shipper,
            consignee=consignee
        )

        return self.output_format

    def get_shipper_prompt(self, shipper):
        prompt = f"""The following is the information of the shipper of a shipment. It contains the name of the shipper, its address and contact information.

        {shipper}

        """
        prompt += r"""
        From that, extract the following JSON:

        {
            "name": "string. Do not put the full address here",
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

    def get_consignee_prompt(self, consignee):
        prompt = f"""The following is the information of the consignee of a shipment. It contains the name of the consignee, its address and contact information.

        {consignee}

        """
        prompt += r"""
        From that, extract the following JSON:

        {
            "name": "string. Do not put the full address here",
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

    def get_pieces(self):
        pieces = []

        pattern = r"\d+ @ \d+ x \d+ x \d+"
        matches = re.findall(pattern, str(self.ret.get('goods_dimentions')))


        for match in matches:
            pc = match.split('@')[0].strip()
            dimensions = match.split('@')[1].strip()
            piece = {
                    "description": self.ret.get("goods_description"),
                    "dimensions": dimensions,
                    "pieces": pc,
                    "weight": None,
                }
            pieces.append(piece)

        return pieces if len(pieces) > 0 else None

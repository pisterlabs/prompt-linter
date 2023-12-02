import asyncio
import openai
import json


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


class OmniLogisticsHWABPostprocessing:

    def __init__(self, ret, output_format):
        self.ret = ret
        self.output_format = output_format

        self.check_for_unsupported_file()

    def check_for_unsupported_file(self):
        if (
                not self.ret.get('scac') or
                self.ret.get('scac', '').lower().strip() != 'omni logistics' or
                not self.ret.get('file_type') or
                self.ret.get('file_type', '').lower().strip() != 'house waybill'
        ):
            raise Exception("Unintended file")

    def parse_output_format(self, **kwargs):
        for i in self.output_format:
            self.output_format[i] = None
        print("<<self.ret<<", self.ret)
        if not self.ret.get('file_type'):
            return

        if 'house waybill' not in self.ret.get('file_type', '').lower():
            return

        self.output_format["scac"] = "OMNG"
        # self.output_format["handling"] = "PUD"  # <-- in the mapping just 'hard codded'
        self.output_format["payment_method"] = "PP"

        self.output_format["OT"] = self.ret.get("OT")
        self.output_format["DT"] = self.ret.get("DT")

        self.output_format["shipment_identifier"] = self.ret.get("HAWB")
        self.output_format["BM"] = self.ret.get("HAWB")
        self.output_format["MA"] = self.ret.get("HAWB")

        self.output_format["SI"] = self.ret.get("SI")

        self.output_format["delivery_date"] = self.ret.get("delivery_date")
        if self.output_format["delivery_date"]:
            self.output_format["delivery_date"] = self.output_format["delivery_date"].replace('On', '').strip()
        self.output_format["delivery_time"] = self.get_time(self.ret.get("delivery_time"))

        self.output_format["ready_date"] = self.ret.get('ETA_date')
        self.output_format["ready_time"] = self.get_time(self.ret.get('ETA_time'))

        self.output_format["shipper"] = kwargs.get('shipper')

        self.output_format["consignee"] = kwargs.get('consignee')

        self.output_format["goods"] = {
            "net_weight": self.ret.get('weight_'),
            "no_of_pieces": self.ret.get('pcs_'),
            "pieces": self.get_pieces_from_table(self.ret['goods_table']),
        }

        self.output_format["carrier"] = {
        "name": None,
        "address": {
            "street": None,
            "city": None,
            "state": None,
            "postal_code": None,
            "country_code": None,
            },
        "contact": {
            "name": None,
            "tel": None,
            "fax": None
            }
        }

    async def run(self):

        consignee_prompt = self.get_address_prompt(self.ret["consignee"])
        shipper_prompt = self.get_address_prompt(self.ret["shipper"])

        resps = await call_llm_multiple_prompts_async(
            [consignee_prompt, shipper_prompt]
        )

        consignee = json.loads(resps[0])
        shipper = json.loads(resps[1])

        self.parse_output_format(
            consignee=consignee,
            shipper=shipper
        )

        return self.output_format

    def get_address_prompt(self, consignee):
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
                "name": "string Located near 'Contact:'",
                "tel": "string Located near 'Phone:'",
                "fax": "string. Might not exist"
            }
        }
        Do not invent information. If the information is not there, leave it as null, unless a default is specified.
        """
        return prompt

    def get_time(self, time):
        result = {
            "time": None,
            "meridien": None,
            "timezone": None,
        }

        if not time:
            return result

        if 'and' in time:
            result["time"] = time.replace('and', '-').replace('between', '').strip()
        elif '-' in time:
            result["time"] = time.strip()
        else:
            result["time"] = time.split()[0]
            result["meridien"] = time.split()[1]

        return result

    def get_pieces_from_table(self, goods_table):
        pieces = []

        if goods_table:
            for i in range(len(goods_table)):
                dict_row = self.parse_1d_table_into_dict(goods_table[i])
                try:
                    pieces_item = {
                        'description': dict_row.get('Description'),
                        'dimensions': ' X '.join([dict_row.get('Length'), dict_row.get('Width'), dict_row.get('Height')]),
                        'weight': dict_row.get('Weight (lb)'),
                        'package_type': dict_row.get('Pkg Type'),
                        'pieces': dict_row.get('Pieces')
                    }
                except:
                    pieces_item = {
                        'description': dict_row.get('Description'),
                        'dimensions': None,
                        'weight': dict_row.get('Weight (lb)'),
                        'package_type': dict_row.get('Pkg Type'),
                        'pieces': dict_row.get('Pieces')
                    }
                finally:
                    # this to ignore completely empty pieces_item
                    res = sum(map(lambda x: 1 if x else 0, pieces_item.values()))

                if res > 0:
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

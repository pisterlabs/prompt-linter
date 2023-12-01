import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from dataclasses import dataclass

# Load our OpenAI API key from our .env file
openai_api_key= os.getenv("OPENAI_API_KEY")
# Create our LLM
llm = OpenAI(
    temperature = 0, 
    model_name = 'text-davinci-003', 
    openai_api_key = openai_api_key,
    max_tokens = 500,
    top_p = 1,  
    frequency_penalty=0,
    presence_penalty=0
)

@dataclass
class Address:
    country: str
    state: str
    city: str
    street: str
    address: str
    full_address: str
    country_cn: str
    state_cn: str
    city_cn: str
    street_cn: str
    address_cn: str
    full_address_cn: str
    latitude: float = None
    longitude: float = None

@dataclass
class AddressService:
    raw_address: str

    def __post_init__(self):
        self.raw_address = self.raw_address.strip()

    def resolveAddress(self) -> Address:
        # Create a template for our prompt
        template = """
        I want to standardize the address below into English and Chinese base on the real addresses in the world, 
        Please help me turn the address below into A-ten column spreadsheet:
        {raw_address}

        Country | State | City | Street | Address | 国家 | 省 |  城市 | 街道 | 详细地址
        For example:
        广东省深圳市福田区福田保税区桃花路28号中宝物流大厦
        China  | Shenzhen | Futian District | Taohua Road | No.28, Zhongbao Logistics Building | 中国 | 深圳 | 福田区 | 桃花路28号 | 中宝物流大厦
        """
        # Create a prompt using our template
        prompt = PromptTemplate(
            input_variables=["raw_address"],
            template=template,
        )
        # call LLM
        final_prompt = prompt.format(raw_address=self.raw_address)
        output = llm(final_prompt)
        # split final_prompt into address_components and trim the space and enter
        address_components = output.split(" | ")
        address_components = [address_component.strip() for address_component in address_components]
        # assign address_components to address_dictionary
        address_dictionary = {
            "country": address_components[0],
            "state": address_components[1],
            "city": address_components[2],
            "street": address_components[3],
            "address": address_components[4],
            "country_cn": address_components[5],
            "state_cn": address_components[6],
            "city_cn": address_components[7],
            "street_cn": address_components[8],
            "address_cn": address_components[9]
        }
        #combine address, street, city, state, country into full_address, seperated by comma
        address_dictionary["full_address"] = address_dictionary["address"] + ", " + address_dictionary["street"] + ", " + address_dictionary["city"] + ", " + address_dictionary["state"] + ", " + address_dictionary["country"]
        # combine country_cn, state_cn, city_cn, street_cn, address_cn into full_address_cn
        address_dictionary["full_address_cn"] = address_dictionary["country_cn"] + address_dictionary["state_cn"] + address_dictionary["city_cn"] + address_dictionary["street_cn"] + address_dictionary["address_cn"]
        # assign dict address_dictionary to Address class
        final_prompt = Address(**address_dictionary)
        return final_prompt


# curr_address = AddressService(raw_address = "广东省东莞市沙田镇东莞保税物流中心临海北路70号大华1号仓")

# # print dict address in json format
# print(curr_address.resolveAddress().__dict__)


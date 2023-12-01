import logging
import random
import numpy as np
from openai import OpenAI, AsyncOpenAI

from generators.entity_generator import EntityGenerator
from llm import LLM
from data_access_object import DataAccessObject
from observer import ObserverManager

class OrgGenerator(LLM, EntityGenerator):
    def __init__(
            self, 
            gpt_client: OpenAI,
            async_gpt_client: AsyncOpenAI, 
            dao: DataAccessObject
            ) -> None:
        LLM.__init__(self, gpt_client, async_gpt_client)
        EntityGenerator.__init__(self)
        self.dao = dao

    def create(self):
        try:
            org_type = self.dao.get_org_type_by_id(2)
            org_name = self.generate_org_attribute(
                'OG_GenOrgName', org_type=org_type
            )
            org_mission = self.generate_org_attribute(
                'OG_GenOrgMission', org_type=org_type, org_name=org_name
            )
            org_desc = self.generate_org_attribute(
                'OG_GenOrgDesc', org_type=org_type, 
                org_name=org_name, org_mission=org_mission
            )

            response = self.dao.insert(
                'organisations',
                name=org_name,
                type=org_type,
                description=org_desc,
                mission=org_mission
            )

            return response.data[0]
        except Exception as e:
            # Properly handle exceptions and log the error
            logging.error(f"Failed to create new organisation: {e}")
            raise

    def update(self, organization):
        # Logic to update an organization
        pass

    def deactivate(self, organization):
        # Logic to deactivate an organization
        pass

    def generate_org_attribute(self, prompt_name: str, **kwargs) -> str:
        prompt = self.dao.get_prompt_by_name(prompt_name).format(**kwargs)
        message = [{"role": "user", "content": prompt}]
        logging.info(f"Prompt: {prompt}")

        org_attribute = self.chat(message, 1.25, 80)
        logging.info(f"Generated attribute: {org_attribute}\n")

        if not org_attribute:
            raise ValueError(f"Failed to generate organisation attribute with prompt: {prompt}")

        return org_attribute

    def create_new_product(self, **kwargs) -> str:
        product = self.dao.get_crypto_product_by_id(1)
        product_id = product['id']
        product_type = product['name']

        product_name = self.generate_product_name(
            org_type=kwargs.get('org_type'), 
            org_name=kwargs.get('org_name'), 
            product_type=product_type
        )

        response = self.dao.insert(
            'products',
            org_id=kwargs.get('org_id'), 
            name=product_name,
            type=product_type
        )

        product_type_ids_with_assets = {1, 2, 3, 4, 5, 8, 10, 14, 16, 23}
        if product_id in product_type_ids_with_assets:
            self.generate_asset(product_name)

        return response.data[0]

    def generate_product_name(self, **kwargs) -> str:
        org_type = kwargs.get('org_type')
        org_name = kwargs.get('org_name')
        product_type = kwargs.get('product_type')

        prompt = self.dao.get_prompt_by_name('OG_GenProductName').format(
            org_type=org_type, org_name=org_name, product_type=product_type
        )

        message = [{"role": "user", "content": prompt}]
        logging.info(f"Prompt: {prompt}")

        product_name = self.chat(message, 1.25, 80)
        logging.info(f"Generated product name: {product_name}\n")

        if not product_name:
            raise ValueError(f"Failed to generate product name with prompt: {prompt}")

        return product_name
        
    def generate_asset(self, product_name: str):
        prompt = self.dao.get_prompt_by_name('OG_GenAssetTicker').format(
            product_name=product_name
        )

        message = [{"role": "user", "content": prompt}]
        logging.info(f"Prompt: {prompt}")
        ticker = self.chat(message, 1.25, 10)
        logging.info(f"Generated ticker: {ticker}")

        ticker = ticker.upper()
        
        cir_supply = self.generate_nice_number(30_000, 1_000_000_000_000_000)
        circ_to_max_ratio = random.randint(1, 100)
        max_supply = self.generate_nice_number(cir_supply, cir_supply * circ_to_max_ratio)

        vc_pre_allocation = random.randint(1, 1000)
        market_cap = 10_000 * vc_pre_allocation

        price = market_cap / cir_supply
        volume_24h = 0
        change_24h = 0

        response = self.dao.insert(
            'assets',
            ticker=ticker, name=product_name, circulating_supply=cir_supply,
            max_supply=max_supply, market_cap=market_cap, price=price,
            volume_24h=volume_24h, change_24h=change_24h
        )

        return response.data[0]
    
    @staticmethod
    def generate_nice_number(min_val, max_val):
        # Generate a random number in the logarithmic scale
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        random_log_val = np.random.uniform(log_min, log_max)

        # Convert back to linear scale and round to nearest power of 10
        nice_number = round(10**random_log_val, -int(np.floor(random_log_val)))

        return nice_number
from langchain.prompts import PromptTemplate
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)

def load_prompt_from_config(config) -> dict:
    """
    Load the prompt for the given config.

    Parameters:
    -------
    config (OmegaConf):
        The configuration object.

    Returns
    -------
    dict: 
        Dictionary containing PromptTemplate objects or an empty dictionary if not found.
    """
    
    chain_type = config.chain.chain_type
    prompts_dict = config.reader.prompt_template.get(chain_type, {})

    if not prompts_dict:
        logger.info("No custom prompt template found. Using langchain default.")
        return {}

    prompt_templates = {}
    for key, value in prompts_dict.items():
        prompt_templates[key] = PromptTemplate(
            template=value.template, 
            input_variables=OmegaConf.to_container(value.input_variables) 
        )

    return prompt_templates

def get_chain_kwargs(config) -> dict:
    return load_prompt_from_config(config)


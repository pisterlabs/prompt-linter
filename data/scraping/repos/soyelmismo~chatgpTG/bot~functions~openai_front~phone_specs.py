from bot.src.utils.constants import ERRFUNC, FUNCNOARG
from bot.src.utils.gen_utils.openai.openai_functions_extraction import openaifunc

from bot.src.apis import smart_gsm
@openaifunc
async def search_smartphone_info(self, model: str) -> str:
    """
    Receives the device name and makes a search in the smart_gsm website returning all the device info.

    Args:
        model (str): only the device model, without extra text.

    Returns:
        str: all the device specifications to be tell to the user
    """
    if model:
        try:
            return await smart_gsm.get_device(self, query = model)
        except Exception: return ERRFUNC
    else: return FUNCNOARG

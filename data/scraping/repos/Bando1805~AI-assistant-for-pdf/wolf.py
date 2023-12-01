from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from API_keys import WOLFRAM_ALPHA_APPID

wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=WOLFRAM_ALPHA_APPID)

print(wolfram.run("What is 2x+5 = -3x + 7?"))
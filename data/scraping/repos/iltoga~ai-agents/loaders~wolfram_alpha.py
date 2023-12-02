import os
import dotenv
import ssl
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

class WolfRamAlpha:
    def __init__(self):
        # this resolves the problem of SSL: CERTIFICATE_VERIFY_FAILED
        ssl._create_default_https_context = ssl._create_unverified_context

        # you must have a .env file in the root directory and add WOLFRAM_ALPHA_APPID as explained here:
        # https://python.langchain.com/en/latest/modules/agents/tools/examples/wolfram_alpha.html
        dotenv.load_dotenv()
        self.wolfram = WolframAlphaAPIWrapper(wolfram_client=None, wolfram_alpha_appid=os.getenv("WOLFRAM_ALPHA_APPID"))
        
    def run(self, query):
        return self.wolfram.run(query)

if __name__ == "__main__":
    domath = WolfRamAlpha()
    res = domath.run("What is 2x+5 = -3x + 7?")
    print(res)
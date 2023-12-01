import os
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

os.environ["WOLFRAM_ALPHA_APPID"] = "6YLRQ4-QW5RR4JQKY"

wolfram = WolframAlphaAPIWrapper()

response = wolfram.run("Impedance of a 10 millihenry inductor at 1000 hertz  ")

# You can print the response or use it as you need
print(response)



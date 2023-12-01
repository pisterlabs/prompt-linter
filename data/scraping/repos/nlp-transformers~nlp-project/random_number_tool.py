from random import *
import numbers
from langchain.tools import Tool
def random_number_generator(query : str):
    # print(query)
    min, max = query.split(",")
    if isinstance(int(min), numbers.Number)==False :
        min = 0

    if isinstance(int(max), numbers.Number)==False:
        max = 100

    r = randint(int(min), int(max))

    return str(r)

random_number_tool = Tool.from_function(
    func = random_number_generator,
    name = "Random number generator",
    description="Use this tool to obtain or generate a random number. Input to this tool should be a comma separated list of minimum and maximum number between which the random number should lie. For example, '1,100' would be the input if you want to generate a random between 1 and 100."
)
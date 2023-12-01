import os
from langchain.tools import tool
from pydantic import BaseModel, Field

from dotenv import dotenv_values

config = dotenv_values(".env")

PATH_LTSPICE = config["LTSPICE_PATH"]
FILE_NAME="rf.net"

class LTSpiceInput(BaseModel):
    netlist: str = Field(description="should be a netlist")

@tool("ltspice", return_direct=True)
def ltspice(netlist: str)->str:
    """Runs ltspice on a given netlist input"""
    parse_netlist(netlist)
    run_ltspice()
    return read_raw_output()

def parse_netlist(input):
    if os.path.isfile(FILE_NAME):
        os.remove(FILE_NAME)
    with open(FILE_NAME, "w") as f:
        f.write(input)

def run_ltspice():
    cmd = f'{PATH_LTSPICE} -ascii -b ./{FILE_NAME}'
    os.system(cmd)

def read_raw_output():
    output = ""
    if os.path.isfile("rf.raw"):
        with open("rf.raw", "r") as f:
            output = f.read()
    else:
        with open("rf.log","r") as f:
            output = f.read()
    return output

if __name__ == '__main__':
    print("Testing the lib functions")
    with open("test.txt") as f:
        print(ltspice(f.read()))



####  Copyright 2023 David Caldwell disco47dave@gmail.com


#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.


import serial, time, re

from typing import Any
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate


template = """
You are an LLM that is acting as a translator between human input and a home automation system.
You will be given a list of commands and arguments that each command can take. 
You will decide based on the human input which command and arguments to pass to the output and format them appropriately.
You will control the lights and fans in various rooms as well as an HVAC thermostat.

Here is a list of the room codes and corresponding rooms:
[
"L" Living Room
"K" Kitchen"
"D" Den
"M" Master Bedroom
"G" Guest Bedroom
"H" Hallway
"B" Bathroom
"C" Carport
]

Here is a list of commands you can use with descriptions of their arguments and output:
[

command: "L"
description:  used to turn lights on or off
input:  a room code and either of "0" for off or "1" for on

command: "F"
description:  used to turn the fan on or off
input:  a room code and either of "0" for off or "1" for on

command: "A"
description:  used to set the thermostat for the HVAC system
input:  the temperature to set

]

If you do not find a suitable command, then respond with "unable"

To format the command string, place a "<" symbol before the command, then the command, then a comma "," and then any arguments as a comma separated list.  Finally include the ">" symbol.  

Here are some examples of formatted command strings:
To turn the hallway light on: <L,H,1>
To turn the kitchen fan off: <F,K,0>
To set the HVAC to 85 degrees <A,85>

Be sure to include only formatted command strings in your response
If you need to send more than one command then each command must be fully formatted in its own set of < and >.
For example to turn off the lights in both the kitchen and bathroom:  <L,K,0><L,B,0>

Here is some current sensor data that you can use if you need it. 
[
Current Temperature {current_temp}
Current HVAC Setting {hvac_setting}
]

Begin!

Human Input:
{input}

"""

class ArduGPTPromptTemplate(StringPromptTemplate):
    template: str
    device: Any    
    
    def format(self, **kwargs) -> str:
        ###  We will get data from the Arduino to use in place of {placeholders} in the prompt
        if self.device is not None:
            ### code to request temperature
            self.device.write(bytes('<R>', 'utf-8'))
            time.sleep(0.1) ### just for demo code
            tempResponse = self.device.readline()     
            ###  Arduino sends back <T,xxx,yyy> where xxx is current temp and yyy is hvac setting       
            match = re.match(br"<T,(\d{1,3}),(\d{1,3})>", tempResponse)
            if match:
                kwargs["current_temp"] = match.group(1).decode('utf-8')
                kwargs["hvac_setting"] = match.group(2).decode('utf-8')
            else:
                kwargs["current_temp"] = "Unknown"
                kwargs["hvac_setting"] = "Unknown"               
        return self.template.format(**kwargs)



class ArduGPT:
    
    def __init__(self, device):
        
        self.exit_flag = False
        self.device = device
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", 
            temperature=0
            )

        prompt = ArduGPTPromptTemplate(
            input_variables=["input"],
            template=template,
            device=self.device,
            )   
        
        self.llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        
        return
    
    
    def getUserInput(self):
        print("\n*******************************\n\nHuman:  ")
        self.human_input = input("")
        if self.human_input == "exit":
            self.exit_flag = True
        return 
    
    def printResponse(self):
        
        if self.ai_output is not None:
            print("\n*******************************\n\nAI:  ")
            print(self.ai_output)
            if 'unable' not in self.ai_output:
                self.device.write(bytes(self.ai_output, 'utf-8'))
                time.sleep(0.1)  ## sleep while we wait for response
                returned = self.device.readline()
                print("\n*******************************\n\nArduino:  ")
                while returned != b'':
                    print(returned)
                    returned = self.device.readline()
            
        
        return 
    
    def getResponse(self):
            
        if self.human_input is not None:
            self.ai_output = self.llm_chain.run(self.human_input)                             
        
        return 
    
    def run(self):
        
        self.getUserInput()
        while not self.exit_flag:
            self.getResponse()
            self.printResponse()
            self.getUserInput()   
        return 
    
arduino = serial.Serial(port='/dev/ttyACM1', baudrate=115200, timeout=0.1)
    

ard = ArduGPT(arduino)
ard.run()

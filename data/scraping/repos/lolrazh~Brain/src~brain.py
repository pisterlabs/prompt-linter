from ghpythonlib.componentbase import dotnetcompiledcomponent as component
import Grasshopper, GhPython
import System
import Rhino
import rhinoscriptsyntax as rs
import clr
import json
import traceback
import System
from System import Array
from System.Text import Encoding
from System.Net import WebRequest, WebHeaderCollection, WebProxy, WebException


# FUNCTION TO SEND API CALL
def send_request(user_prompt, Description, API_Key):
    plugin_context = "You are GrasshopperAI, an AI tool working within Grasshopper to manipulate Grasshopper scripts through natural language. You will be given a description of the entire script and data about the sliders in the format <SLIDER_NAME> - <SLIDER_VALUE>. After understanding the prompt, you have to understand what slider to change and what the new slider value would be. To do that, all you have to do is call a function that I have already programmed which is change_slider_value(str(<SLIDER_NAME>), <NEW_SLIDER_VALUE>). Remember to respond only with the function and enclose the name in an str() function." # system prompt for gpt
    url = "https://api.openai.com/v1/chat/completions"  # reference api url
    data = { # prepare data for request
        "model": "gpt-3.5-turbo-16k", # gpt model, gpt-3.5 is much cheaper to run than gpt4
        "messages": [
            {
                "role": "system",
                "content": str(plugin_context) + "\n" + "\n" + str(Description)  # system prompt is set up here
            },
            {
                "role": "user",
                "content": user_prompt  # user prompt goes here
            }
        ],
        "temperature": 0.5,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    data_string = json.dumps(data)
    data_bytes = Encoding.UTF8.GetBytes(data_string)
    request = WebRequest.Create(url) # create a web request
    request.Method = "POST"
    request.ContentType = "application/json"
    request.Headers.Add("Authorization", "Bearer " + str(API_Key)) # authentication
    request_stream = request.GetRequestStream() # write request to data stream
    request_stream.Write(Array[System.Byte](data_bytes), 0, len(data_bytes))
    request_stream.Close()
    try:
        response = request.GetResponse()
        response_stream = response.GetResponseStream()
        response_reader = System.IO.StreamReader(response_stream)
        response_text = response_reader.ReadToEnd()

        response_json = json.loads(response_text) # get response from openai
        return response_json

    except WebException as e: # error handling
        display_error(str(e))
        return None


# DEFINE USER PROMPT HERE
def construct_prompt(param_data):
    sliders = str(param_data)[1:-1] # get slider data
    user_prompt = rs.StringBox(message="Enter prompt", title="Brain")
    if user_prompt is None:  # check if user cancelled the input
        return None
    prompt = str(sliders) + "\n" + "\n" + str(user_prompt) # construct prompt
    return prompt


class Brain(component): # set up component class
    def __new__(cls):
        instance = Grasshopper.Kernel.GH_Component.__new__(cls,
            "Brain", "Brain", """Uses GPT to manipulate Grasshopper scripts through natural language.""", "Params", "Util") # component info
        return instance
    
    def get_ComponentGuid(self):
        return System.Guid("bc7f32af-9b9c-4cf6-938a-21b24702c0da")
    
    def SetUpParam(self, p, name, nickname, description): # set up component parameters
        p.Name = name
        p.NickName = nickname
        p.Description = description
        p.Optional = True
    
    def RegisterInputParams(self, pManager):
        p = Grasshopper.Kernel.Parameters.Param_Number()
        self.SetUpParam(p, "Parameters", "P", "Parameters to control")
        p.Access = Grasshopper.Kernel.GH_ParamAccess.list
        self.Params.Input.Add(p)
        
        p = Grasshopper.Kernel.Parameters.Param_String()
        self.SetUpParam(p, "API_Key", "K", "OpenAI API Key")
        p.Access = Grasshopper.Kernel.GH_ParamAccess.item
        self.Params.Input.Add(p)
        
        p = Grasshopper.Kernel.Parameters.Param_String()
        self.SetUpParam(p, "Description", "D", "A brief description of the GH definition")
        p.Access = Grasshopper.Kernel.GH_ParamAccess.item
        self.Params.Input.Add(p)
        
        p = Grasshopper.Kernel.Parameters.Param_Boolean()
        self.SetUpParam(p, "Trigger", "R", "Runs the component")
        p.Access = Grasshopper.Kernel.GH_ParamAccess.item
        self.Params.Input.Add(p)
        
    def RegisterOutputParams(self, pManager):
        pass    

    def SolveInstance(self, DA):
        p0 = self.marshal.GetInput(DA, 0)
        p1 = self.marshal.GetInput(DA, 1)
        p2 = self.marshal.GetInput(DA, 2)
        p3 = self.marshal.GetInput(DA, 3)
        result = self.RunScript(p0, p1, p2, p3)

    def get_Internal_Icon_24x24(self): # component icon
        o = "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAFiUAABYlAUlSJPAAAAI6SURBVEhLvdVJyE5RHMfxa54zZYqQspAsJDZkLgtZGBYoUli9xcKwYMdKWFmYsjBmWGBhxUqRjQgrJXNKUobM0/d77jk99z3v875vj0d+9em553Tv+d9znnPvLRpMn/j7TzIBY9AVk3AFvyt+4TrGoaGsxWM4gO7jG37iAtZhYzy20FtMQ6fpi4vwouc4hZOx/QJTkWcxPsEbGmhHR0mD7URvO8guOIsZoVWme9QPw7ABXteCdjMbnrQ3tMpMgX2HQ6sopuMdLPgFLpm+w/NeYyHqxkE8aVRoFUUX3MQHjLCDOLiDPcFRrMECWNhZW9QxDsEZtspdeHHKSnjyptAql+8HVoVW/YzEPnjdETuquQOna9yWz2Dbfe9SfcV7WHg1lkbL4u9wGGd+BhaZZUfKHtjpfjdX4Yx8Fs7BYs6guuZV15AyFC7XwdCKGQ//vMuhVT48nuRWddAtGILJmIM3yIusQIpb91Z5WMtxeOL80Kr98bdDq3XcmhauFngInyXjLG+Uh7UMhmv9CP4PnvwRB5BnIqqDJ9vQAxY4hjbZDk9cH1rlVvQ14fJU41K6pHmBl3DtPV6ENvHOH8Dpb8VyWOAS5sIH8iy8w0E4gbyITqPdzIQDeIfuHPmnfYZLaHszjFs2H/wVOn0vdUOvqGc0AKMxFsa+e8gLuEz90XTSSy73FOll+dfxYfJO6xU4j6azH/UG1zw0Fb9e6c2Z8/vRdNI+r3LndPjBaSR+xZZgB3bDb3TaWf87RfEHQBfFoHsNd3QAAAAASUVORK5CYII="
        return System.Drawing.Bitmap(System.IO.MemoryStream(System.Convert.FromBase64String(o)))

    def RunScript(self, Parameters, API_Key, Description, Trigger):
        # FUNCTION FOR ERROR HANDLING
        def display_error(message): # display error box
            rs.MessageBox(message, 16, "Error")
        # FUNCTION TO CHANGE SLIDER VALUE
        def change_slider_value(nickname, value):
            ghdoc = self.OnPingDocument()
            if not ghdoc:
                display_error("Error accessing Grasshopper document.")
                return
            for obj in ghdoc.Objects:
                if isinstance(obj, Grasshopper.Kernel.Special.GH_NumberSlider) and obj.NickName == nickname:
                    if obj.Slider.Minimum <= value <= obj.Slider.Maximum: # check slider domains
                        obj.SetSliderValue(value)
                        obj.ExpireSolution(True)
                    else:
                        display_error("{} is out of range for '{}'.".format(value, nickname)) # error handling

        # START RUNSCRIPT
        ghdoc = self.OnPingDocument()
        if not ghdoc:
            display_error("Error accessing Grasshopper document.") # error handling
        else:
            input_params = self.Params.Input[0].Sources if self.Params.Input else [] # get component inputs
            param_data = [(param.NickName, param.CurrentValue) for param in input_params if isinstance(param, Grasshopper.Kernel.Special.GH_NumberSlider)] # get slider data
        if param_data:
            if Trigger:
                prompt = construct_prompt(param_data) # construct user prompt
                if prompt is not None: # avoid running when user hits cancel
                    response = send_request(prompt, Description, API_Key) # send api call
                    if response:  # check if response is not None
                        function = response['choices'][0]['message']['content'] # pick out the answer from the response json
                        try:
                            eval(str(function)) # call function
                        except Exception as e:  # catch exceptions raised during the evaluation
                            display_error("Error evaluating function: {}".format(str(e)))
                    else:
                        display_error("Failed to get a valid response from the OpenAI API.")
                else:
                    pass


import GhPython
import System

class AssemblyInfo(GhPython.Assemblies.PythonAssemblyInfo):
    def get_AssemblyName(self):
        return "Brain"
    
    def get_AssemblyDescription(self):
        return """"""

    def get_AssemblyVersion(self):
        return "0.1"

    def get_AuthorName(self):
        return "Sandheep Rajkumar" # haha me hahaha
    
    def get_Id(self):
        return System.Guid("88adbca2-bd32-4edc-9be2-9ebf3faaf3fb")

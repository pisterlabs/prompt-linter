from langchain.tools import BaseTool

from webcrawl import *
from .Nl2ModelChain import ModelicaFlareChain
from .Nl2Modelica import ModelObject
from pydantic import Field
from io import StringIO
import logging
import matplotlib.pyplot as plt
import io
import sys
from OMPython import ModelicaSystem

import ast
__all__ = [
    "LookupTool",
    "WriteTool",
#    "CheckTool",
#    "SetInputsTool",
#    "SetParametersTool",
#    "SetSimOptionsTool",
    "SimulateTool",
    "ResultsTool",
    "Nl2ModelTools",
    ]
class OutputInterceptor(io.StringIO):
    def __init__(self):
        super().__init__()

    def getvalue(self):
        result = super().getvalue()
        self.truncate(0)
        self.seek(0)
        return result
class LookupTool(BaseTool):
    modelica_model: ModelObject
    def __init__(self, modelica_model, **kwargs):
        super().__init__(modelica_model=modelica_model, **kwargs)
        self.modelica_model = modelica_model
    name = "SearchDocs"
    description = "Submits a detailed and verbose natural language query to fetch and present relevant information from the modelica and modelica documentation. Use if more reference required."

    return_direct = False
    return_intermediate_steps = True
    def _run(self, query: str) -> str:
        """Use the tool synchronously."""
        kwargs = {
            "question":query,
            "chat_history": f"{self.modelica_model.lookup_mem.buffer}"
            }

        response = self.modelica_model.lookup_chain.run(**kwargs)
                    # Search for the 'answer' part
                    
        if response:
            logging.info(f"LookupTool modelica_model: {id(self.modelica_model)}")

            self.modelica_model.modelica_context = response
            logging.info(f"Lookup Context: {self.modelica_model.modelica_context}")
            return f"{response}"
        else:
            return "No answer found in the response."
        
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        kwargs = {
            "question":query,
            "chat_history": f"{self.modelica_model.lookup_mem.buffer}",
            }

        response = await self.modelica_model.lookup_chain.arun(**kwargs)#callbacks=self.modelica_model.lookup_chain.callbacks,
                    # Search for the 'answer' part
                    
        if response:
            logging.info(f"LookupTool modelica_model: {id(self.modelica_model)}")

            self.modelica_model.modelica_context = response
            logging.info(f"Lookup Context: {self.modelica_model.modelica_context}")
            return "The relevant context was added to memory.  Use Nl2Code next."
        else:
            return "No relevant context was found.  Attempt the solution using Nl2Code without it."
    
class WriteTool(BaseTool):
    modelica_chain: ModelicaFlareChain
    def __init__(self,modelica_chain, **kwargs):
        super().__init__(
            modelica_chain=modelica_chain,
            **kwargs
            )
        self.modelica_chain = modelica_chain
        logging.info(f"WriteTool init modelica_chain: {id(self.modelica_chain)}")

    name = "Nl2model"
    description = """
    Provides a verbose natural language (nl) description identifying the key components, parameters, and their interactions 
    to be modeled in Modelica. The context retrieved from SearchDocs are also used to create a model.
    """
    MAX_RUNS = 3

    return_direct = False
    return_intermediate_steps = True

    def extract_model_or_class_name(self):
        pattern = re.compile(r'(model|class)\s+([^\s]+)')
        match = pattern.search(self.modelica_chain.modelica_model.code)
        if match:
            return match.group(2)
        else:
            return None
    def write_code(self,query):
        kwargs = {
            "user_input": query,
            }
        for iter in range(0,self.MAX_RUNS):
            response = self.modelica_chain.run(**kwargs)
            if response:
                self.modelica_chain.modelica_model.code = response
                self.modelica_chain.modelica_model.model_name = self.extract_model_or_class_name()
                if not self.modelica_chain.modelica_model.model_name or not self.modelica_chain.modelica_model.code:
                    self.modelica_chain.modelica_model.modelica_context += "\n Model or class name not found in code: {response}."
                else:
                    # Append the response to a file
                    with open(f"./{self.modelica_chain.modelica_model.model_file}", 'w') as f:
                        f.write(self.modelica_chain.modelica_model.code)

                    # Create an interceptor
                    interceptor = OutputInterceptor()
                    # Save the original stdout and stderr
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    errorlog = None
                    try:
                        # Redirect stdout and stderr to the interceptor
                        sys.stdout = interceptor
                        sys.stderr = interceptor
                        # code that generates stderr output
                        self.modelica_chain.modelica_model.modelica_system=ModelicaSystem(
                            f"./{self.modelica_chain.modelica_model.model_file}",
                            self.modelica_chain.modelica_model.model_name,
                            ["Modelica"],
                            )
                        self.modelica_chain.modelica_model.modelica_system.buildModel()
                    except Exception as e:
                        # Handle any Python exceptions here
                        return(f"{e}")
                    finally:
                        # Restore the original stdout and stderr
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr

                    # Get a string containing the stderr output
                    #if errorlog:
                    #    output = errorlog
                    #else:
                        # Get the intercepted output
                        output = interceptor.getvalue()

                    logging.info(f"flare2 modelica_model: {id(self.modelica_chain.modelica_model)}, {output}")

                    if output:
                        self.modelica_chain.modelica_model.modelica_context += f"\n\nPrevious Response Attempt: {self.modelica_chain.modelica_model.code}\nBuild model output: {output}"
                    else:
                        #self.modelica_chain.modelica_model.modelica_input = ""
                        self.modelica_chain.modelica_model.quantities = self.modelica_chain.modelica_model.modelica_system.getQuantities()
                        self.modelica_chain.modelica_model.continuous = self.modelica_chain.modelica_model.modelica_system.getContinuous()
                        self.modelica_chain.modelica_model.inputs     = self.modelica_chain.modelica_model.modelica_system.getInputs()
                        self.modelica_chain.modelica_model.outputs    = self.modelica_chain.modelica_model.modelica_system.getOutputs()
                        self.modelica_chain.modelica_model.parameters = self.modelica_chain.modelica_model.modelica_system.getParameters()
                        self.modelica_chain.modelica_model.simOptions = self.modelica_chain.modelica_model.modelica_system.getSimulationOptions()
                        self.modelica_chain.modelica_model.solutions  = self.modelica_chain.modelica_model.modelica_system.getSolutions()
                        return f"{self.modelica_chain.modelica_model.model_name} successfully written/loaded. Use RunSim next."
                    
            else:
                return "Something went wrong. No response found."
        
        return "Repeated attempts at generating working code fail.  Try a different input."
        
            
    def _run(self, query: str) -> str:
        """Use the tool synchronously."""
        
        logging.info(f"WriteTool modelica_model: {id(self.modelica_chain.modelica_model)}")
        logging.info(f"WriteTool modelica_chain: {id(self.modelica_chain)}")
        response = self.write_code(query)
        return f"{response}"
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        logging.info(f"WriteTool modelica_model: {id(self.modelica_chain.modelica_model)}")
        logging.info(f"WriteTool modelica_chain: {id(self.modelica_chain)}")
        response = self.write_code(query)
        return f"{response}"

class CheckTool(BaseTool):
    modelica_model: ModelObject  # Assuming model is an instance of a class with appropriate methods
    def __init__(self, modelica_model, **kwargs):
        super().__init__(modelica_model=modelica_model, **kwargs)
        self.modelica_model = modelica_model
    name = "CheckState"
    description = """
    Fetches and returns a current state of the model. Input should be one of: 'quantities', 'continuous', 'inputs', 'outputs', 'parameters', 'simOptions', 'solutions', 'code'.
    """

    return_direct = False
    return_intermediate_steps = True
    

    def _run(self, query: str = None) -> str:
        """Use the tool synchronously."""
        list_names = ['quantities', 'continuous', 'inputs', 'outputs', 'parameters', 'simOptions', 'solutions', 'code']
        if query and query in list_names:
            response = self.modelica_model.get_value(query)
            return response
        else:
            return f"Invalid name. Must be one of: {list_names}"
            #responses = {name: self.modelica_model.get_value(name) #for name in list_names}
            #return responses
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        list_names = ['quantities', 'continuous', 'inputs', 'outputs', 'parameters', 'simOptions', 'solutions', 'code']
        if query and query in list_names:
            response = self.modelica_model.get_value(query)
            return response
        else:
            return f"Invalid name. Must be one of: {list_names}"
            #responses = {name: self.modelica_model.get_value(name) #for name in list_names}
            #return responses

class SetInputsTool(BaseTool):
    modelica_model: ModelObject
    def __init__(self, modelica_model, **kwargs):
        super().__init__(modelica_model=modelica_model, **kwargs)
        self.modelica_model = modelica_model
    name = "DefineModelInputs"
    description = """
    Defines model inputs using key-value pairs ('key=value'). This tool can accept either a single string or a list of strings (e.g., "cAi=1" or ["cAi=1", "Ti=2"]). Brackets should be included in the input when providing a list of parameters.
    """

    return_direct = False
    return_intermediate_steps = True
    
    def parse_input(self, input_str: str):
        try:
            # remove quotes if present
            input_str = input_str.strip('"\'')
            # if input_str is a single string, wrap it in a list for consistency
            if isinstance(input_str, str):
                input_str = [input_str]
            parsed_input = [self.parse_key_value_string(i) for i in input_str]
            return parsed_input
        except ValueError as e:
            raise ValueError(f"Invalid input format: {e}")

    def parse_key_value_string(self, kv_str: str):
        if '=' not in kv_str:
            raise ValueError(f"Invalid key-value pair: {kv_str}")
        key, value = kv_str.split('=', 1)
        return key, value

    def _run(self, query: str) -> str:
        """Use the tool synchronously."""
        try:
            inputs_list = self.parse_input(query)
            self.modelica_model.modelica_system.setInputs(inputs_list)
            self.modelica_model.inputs = self.modelica_model.modelica_system.getInputs()
            # Perform further processing or return the results
            return f"Successfully set inputs to: {self.modelica_model.inputs}."
        except ValueError as e:
            return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        try:
            inputs_list = self.parse_input(query)
            self.modelica_model.modelica_system.setInputs(inputs_list)
            self.modelica_model.inputs = self.modelica_model.modelica_system.getInputs()
            # Perform further processing or return the results
            return f"Successfully set inputs to: {self.modelica_model.inputs}."
        except ValueError as e:
            return f"Error: {e}"

class SetParametersTool(BaseTool):
    modelica_model: ModelObject
    def __init__(self, modelica_model, **kwargs):
        super().__init__(modelica_model=modelica_model, **kwargs)
        self.modelica_model = modelica_model
    name = "SetParameters"
    description = """
    Defines model parameters using key-value pairs ('key=value'). This tool can accept either a single string or a list of strings (e.g., "cAi=1" or ["cAi=1", "Ti=2"]). Brackets should be included in the input when providing a list of parameters.
    """

    return_direct = False
    return_intermediate_steps = True
    
    def parse_input(self, input_str: str):
        try:
            # remove quotes if present
            input_str = input_str.strip('"\'')
            # if input_str is a single string, wrap it in a list for consistency
            if isinstance(input_str, str):
                input_str = [input_str]
            parsed_input = [self.parse_key_value_string(i) for i in input_str]
            return parsed_input
        except ValueError as e:
            raise ValueError(f"Invalid input format: {e}")

    def parse_key_value_string(self, kv_str: str):
        if '=' not in kv_str:
            raise ValueError(f"Invalid key-value pair: {kv_str}")
        key, value = kv_str.split('=', 1)
        return key, value

    def _run(self, query: str) -> str:
        """Use the tool synchronously."""
        try:
            inputs_list = self.parse_input(query)
            self.modelica_model.modelica_system.setInputs(inputs_list)
            self.modelica_model.inputs = self.modelica_model.modelica_system.getInputs()
            # Perform further processing or return the results
            return f"Successfully set parameters to: {self.modelica_model.inputs}."
        except ValueError as e:
            return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        try:
            inputs_list = self.parse_input(query)
            self.modelica_model.modelica_system.setInputs(inputs_list)
            self.modelica_model.inputs = self.modelica_model.modelica_system.getInputs()
            # Perform further processing or return the results
            return f"Successfully set parameters to: {self.modelica_model.inputs}."
        except ValueError as e:
            return f"Error: {e}"

class SetSimOptionsTool(BaseTool):
    modelica_model: ModelObject
    def __init__(self, modelica_model, **kwargs):
        super().__init__(modelica_model=modelica_model, **kwargs)
        self.modelica_model = modelica_model
    name = "ConfigureSimulationOptions"
    description = """
    Sets simulation options using key-value pairs ('key=value'). This tool can accept either a single string or a list of strings (e.g., "stopTime=2.0" or ["stopTime=2.0", "tolerance=1e-08"]). Brackets should be included in the input when providing a list of parameters.
    """

    return_direct = False
    return_intermediate_steps = True
   

    def parse_input(self, input_str: str):
        try:
            # remove quotes if present
            input_str = input_str.strip('"\'')
            # if input_str is a single string, wrap it in a list for consistency
            if isinstance(input_str, str):
                input_str = [input_str]
            parsed_input = [self.parse_key_value_string(i) for i in input_str]
            return parsed_input
        except ValueError as e:
            raise ValueError(f"Invalid input format: {e}")

    def parse_key_value_string(self, kv_str: str):
        if '=' not in kv_str:
            raise ValueError(f"Invalid key-value pair: {kv_str}")
        key, value = kv_str.split('=', 1)
        return key, value

    def _run(self, query: str) -> str:
        """Use the tool synchronously."""
        try:
            options_list = self.parse_input(query)
            self.modelica_model.modelica_system.setSimulationOptions(options_list)
            self.modelica_model.simOptions = self.modelica_model.modelica_system.getSimulationOptions()
            # Perform further processing or return the results
            return f"Successfully set simulation options to: {self.modelica_model.simOptions}."
        except ValueError as e:
            return f"Error: {e}"


    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        try:
            options_list = self.parse_input(query)
            self.modelica_model.modelica_system.setSimulationOptions(options_list)
            self.modelica_model.simOptions = self.modelica_model.modelica_system.getSimulationOptions()
            # Perform further processing or return the results
            return f"Successfully set simulation options to: {self.modelica_model.simOptions}."
        except ValueError as e:
            return f"Error: {e}"

class SimulateTool(BaseTool):
    modelica_model: ModelObject
    def __init__(self, modelica_model, **kwargs):
        super().__init__(modelica_model=modelica_model, **kwargs)
        self.modelica_model = modelica_model
    name = "RunSim"
    description = "Executes the simulation of the loaded model using the currently defined parameters and options. Input not required."

    return_direct = False
    return_intermediate_steps = True
    

    async def _run(self, query: str) -> str:
        """Use the tool synchronously."""
        # Temporarily redirect stderr
        # Create an interceptor
        interceptor = OutputInterceptor()

        # Save the original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            # Redirect stdout and stderr to the interceptor
            sys.stdout = interceptor
            sys.stderr = interceptor

            # Perform further processing or return the results
            self.modelica_model.modelica_system.simulate(resultfile=self.modelica_model.results_file)
            self.modelica_model.solutions         = self.modelica_model.modelica_system.getSolutions()
            self.modelica_model.outputs           = self.modelica_model.modelica_system.getOutputs()
        except Exception as e:
            # Handle any Python exceptions here
            return e
        finally:
            # Restore the original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        # Get the intercepted output
        output = interceptor.getvalue()
        if (output):
            return f"Simulation ran with the following output: {output}."
        else:
            return f"Simulation ran successfully.\n Solutions: {self.modelica_model.solutions}. Use PlotResults next."
        

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        # Temporarily redirect stderr
        # Create an interceptor
        interceptor = OutputInterceptor()

        # Save the original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            # Redirect stdout and stderr to the interceptor
            sys.stdout = interceptor
            sys.stderr = interceptor

            # Perform further processing or return the results
            self.modelica_model.modelica_system.simulate(resultfile=self.modelica_model.results_file)
            self.modelica_model.solutions         = self.modelica_model.modelica_system.getSolutions()
            self.modelica_model.outputs           = self.modelica_model.modelica_system.getOutputs()
        except Exception as e:
            # Handle any Python exceptions here
            return e
        finally:
            # Restore the original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        # Get the intercepted output
        output = interceptor.getvalue()
        if (output):
            return f"Simulation ran with the following output: {output}."
        else:
            return f"Simulation ran successfully.\n Solutions: {self.modelica_model.solutions}. Use PlotResults next."


class ResultsTool(BaseTool):
    modelica_model: ModelObject

    def __init__(self, modelica_model, **kwargs):
        super().__init__(modelica_model=modelica_model, **kwargs)
        self.modelica_model = modelica_model

    name = "PlotResults"
    description = "Retrieves two solutions from the model's output to plot against each other and display to the user. The input should be two solution names separated by a comma (e.g. \"x,y\")."

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        # parse query, retrieve solutions, and plot
        try:
            solution1, solution2 = query.split(",")
            solution1 = solution1.strip()
            solution2 = solution2.strip()

            logging.info(f"solution1: '{solution1}', solution2: '{solution2}', solutions: {self.modelica_model.solutions}")  # Debug print

            if solution1 not in self.modelica_model.solutions or solution2 not in self.modelica_model.solutions:
                logging.info(f"solutions: {self.modelica_model.solutions}")  # Debug print
                return f"One or both solutions not found in model solutions {self.modelica_model.solutions}."

            solution1_index = self.modelica_model.solutions.index(solution1)
            solution2_index = self.modelica_model.solutions.index(solution2)
            solution1_data = self.modelica_model.modelica_system.getSolutions(self.modelica_model.solutions[solution1_index])[0]
            solution2_data = self.modelica_model.modelica_system.getSolutions(self.modelica_model.solutions[solution2_index])[0]
            #logging.info(f"{self.modelica_model.solutions[solution1_index]}{solution1_data}")
            #logging.info(f"{self.modelica_model.solutions[solution2_index]}{solution2_data}")
            fig, ax = plt.subplots()
            ax.plot(solution1_data, solution2_data)
            ax.set_xlabel(solution1)
            ax.set_ylabel(solution2)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            await self.modelica_model.lookup_chain.callbacks[0].on_results(buf)
        except Exception as e:
            return str(e)

        return f"Results have been plotted successfully and sent. Generate the final response now."


    def _run(self, query: str) -> str:
        """Use the tool asynchronously."""
        return "501 error. Not implemented."


def Nl2ModelTools(
        modelica_chain,
        ):
    tools = [
        LookupTool(modelica_model=modelica_chain.modelica_model),
        WriteTool(modelica_chain=modelica_chain),
        #CheckTool(),
        #SetInputsTool(),
        #SetParametersTool(),
        #SetSimOptionsTool(),
        SimulateTool(modelica_model=modelica_chain.modelica_model),
        ResultsTool(modelica_model=modelica_chain.modelica_model)
    ]
    return tools
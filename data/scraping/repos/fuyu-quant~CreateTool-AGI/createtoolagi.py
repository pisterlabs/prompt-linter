from langchain.llms import OpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent

import warnings
warnings.filterwarnings('ignore')

class CreateToolAGI():
    def __init__(
        self,
        model_name = 'gpt-4'
        ):
        self.model_name = model_name

    
    def _generalize(self, input_prompt_):
        print("> Generalize the input task.")

        generalize_llm = OpenAI(temperature=0,model_name = self.model_name)

        prompt = """
        Please generalize the following sentences by replacing any numerical or other parts of the sentences with letters.
        Output is only the text after rewriting.

        ------------
        {input}
        ------------
        """.format(input = input_prompt_)

        responese = generalize_llm(prompt)

        print('\033[32m' + f'Generalized Task：{responese}\n' + '\033[0m')

        return responese
        

    def _decide(self, input_prompt_, tools_):
        print("> Determine if you should make a tool.")
        
        decide_llm = OpenAI(temperature=0,model_name = self.model_name)

        description_list = ''
        for tool_ in tools_:
            description = tool_.description
            description_list += description + ','
        description_list = '[' + description_list + ']'

        prompt = """
        You are an agent that determines if the input task is executable. 
        All you can do is to be included in {exec_list}. 
        Answer "Yes." if you can perform the task, or "No." if you cannot.
        
        -----------
        The entered task is:{input}
        -----------
        """.format(exec_list = description_list, input = input_prompt_)

        responese = decide_llm(prompt)

        if responese == "Yes.":
            print('\033[32m' + 'I do not need to create a tool to run it.\n' + '\033[0m')

        elif responese == "No.":
            print('\033[32m' + 'I must create the tool before executing.\n' + '\033[0m')


        return responese


    def _tool_make(self, input_prompt_, folder_path_, refer_code_ = None):
        print("> Create a tool.")

        tool_llm = OpenAI(temperature=0, model_name = self.model_name)


        create_prompt = """
        Please create your code in compliance with all of the following conditions.
        ・Create a python class that can execute {input} with a single string as input.
        ・Output should be code only.
        ・The following code was created with the input "multiply two numbers". Please create it like this code.
        ・Do not enclose the output in ``python ``` or the like.
        ・from langchain.tools import BaseTool must be written.
        ・Class must inherit from BaseTool.
        ・If you have previously created code that failed to execute, please refer to it as well.
        ・Here is the code I created previously: {code}
        
        ------------------
        from langchain.tools import BaseTool

        class MultiplicationTool(BaseTool):
            name = "MultiplicationTool"
            description = "used for multiplication. The input is two numbers. For example, if you want to multiply 1 by 2, the input is '1,2'."

            def _run(self, query: str) -> str:
                "Use the tool."
                a, b = query.split(",")
                c = int(a) * int(b)
                result = c

            return result 

            async def _arun(self, query: str) -> str:
                "Use the tool asynchronously."
                raise NotImplementedError("MultiplicationTool does not support async")
        ------------------
        """.format(input = input_prompt_, code = refer_code_)

        code = tool_llm(create_prompt)



        name_prompt = """
        Extract the class name from the following code.
        The final output is only the extracted content.

        ------------------
        {code}
        ------------------
        """.format(code = code)

        name = tool_llm(name_prompt)
        print('\033[32m' + 'Completed!')
        print('Created tool name：' + name + '\n' + '\033[0m')

        # Save to File
        if folder_path_ != None:
            with open(folder_path_ + f'{name}.py', mode='w') as file:
                file.write(code)
        

        return name, code


    def _execute(self,input_prompt_, tools_):
        excute_llm = OpenAI(temperature=0, model_name = self.model_name)

        agent = initialize_agent(tools_, excute_llm, agent="zero-shot-react-description", verbose=True)

        agent.run(input_prompt_)

        return


    def run(self, input_prompt_, tools_, folder_path_ = None):

        generalized_task = self._generalize(input_prompt_)
    
        output = self._decide(generalized_task, tools_)

        if output == "Yes.":
            self._execute(input_prompt_, tools_)

        elif output == "No.":
            tools_ = []
            # Repeat until successful.
            count = 0
            refer_tool_code = None
            while count < 5:
                count += 1
                print(f'Try:{count}')
                try:
                    new_tool_name, new_tool_code = self._tool_make(generalized_task, folder_path_, refer_tool_code)
                    
                    # If the execution fails, it is referenced in the next loop.
                    refer_tool_code = new_tool_code
                    
                    new_tool_code = new_tool_code + '\n' + f'new_tool = {new_tool_name}()'
                    
                    exec(new_tool_code, globals())
                    
                    tools_.append(new_tool)
            
                    self._execute(input_prompt_, tools_)

                    break
                except Exception as e:
                    print('\033[32m' + f"Error occurred: {e}" + '\n' + '\033[0m')
                    
            if count >= 5:
                print('\033[32m' + "Reached the maximum number of tries." + '\033[0m')
        
        return 
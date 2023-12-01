from langchain.llms import OpenAI
import re


class Creator():
    def __init__(
        self,
        create_model,
        ):
        self.create_model = create_model

    def run(self, generalized_input, search_result, created_tool_code):
        print("> Create a tool.")

        related_tools = ''
        for i in range(len(search_result)):
            disc = search_result[i].payload['discription']
            code = search_result[i].payload['code']
            related_tools += 'discription:' + '\n' + disc +'\n' + 'code:' +'\n' + code + '\n' + '------------------' + '\n'



        create_prompt = """
        Please create your code in compliance with all of the following conditions.
        ・Create a python class that can take a single string as input and perform the following.
        ------------------
        {input_}
        ------------------
        ・Output should be code only.
        ・Do not enclose the output in ``python ``` or the like.
        ・from langchain.tools import BaseTool must be written.
        ・Class must inherit from BaseTool.
        ・If you have previously created code that failed to execute, please refer to it as well.
        ・Here is the code I created previously: {created_tool_code_}
        ・The following code was created with the input "multiply two numbers". Please create it like this code.
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
        ・The following is a code that may be similar. Please refer to them if they exist.
        ------------------
        {related_tools_}
        """.format(
            input_ = generalized_input, 
            created_tool_code_ = created_tool_code, 
            related_tools_ = related_tools
            )

        #print(create_prompt)

        code = self.create_model(create_prompt)

        tool_name = re.search(r'name = "(.*?)"', code).group(1)


        print('\033[32m' + 'Completed!')
        print('Created tool name：' + tool_name + '\n' + '\033[0m')

        # Save to File
        #if folder_path_ != None:
        #    with open(folder_path_ + f'{name}.py', mode='w') as file:
        #        file.write(code)

        return code
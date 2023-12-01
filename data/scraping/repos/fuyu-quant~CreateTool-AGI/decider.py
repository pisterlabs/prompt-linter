from langchain.llms import OpenAI


class Decider():
    def __init__(
        self,
        base_model,
        ):
        self.base_model = base_model

    def run(self, generalized_input, search_result):
        print("> Decide if the task is executable.")
        
        executable_tasks = ''
        
        for i in range(len(search_result)):
            desc = search_result[i].payload['description']
            code = search_result[i].payload['code']
            executable_tasks += 'description:' + '\n' + desc +'\n' + 'code:' +'\n' + code + '\n' + '------------------' + '\n'

        prompt = """
        Please answer the following questions in accordance with the following conditions.
        ・Determine if the entered task is executable.
        ・Answer "Yes." if you can perform the task, or "No." if you cannot.
        ・Please make sure to answer with "Yes." or "No.". Otherwise, do not include them in the output.
        ・You execute tasks using the following tools. Please refer to the descriptions of the tools and the code to determine whether the task can be executed or not.
        ・The task you are to perform is {input_}.


        The following tools are available to you.
        -----------
        {executable_tasks_}
        
        """.format(input_ = generalized_input, executable_tasks_ = executable_tasks)

        responese = self.base_model(prompt)

        print(f'Response：{responese}')

        if responese == "Yes.":
            print('\033[32m' + 'I do not need to create a tool to run it.\n' + '\033[0m')

        #else:
        elif responese == "No.":
            print('\033[32m' + 'I must create the tool before executing.\n' + '\033[0m')


        return responese
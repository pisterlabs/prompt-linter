import openai
class apikey:
    def __init__(self, api_key):
        openai.api_key  = api_key
        #self.api_key = api_key
        self.co = None
        self.da= None
    def find_all(self, da, qu):
        query = f"""with reference to this given paragraph- "{da}", Consider below instructions for writing one line regex code for it - (this para is already stored into "da" variable. So, don't use the paragraph into your code and use "da" variable instead and write full single line python regex code for the given query - "{qu}", store the output into "result" variable.)"""

        coderr = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[
            {"role": "system",
             "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-03-02"},
            {"role": "user", "content": query}
        ])

        cod = coderr["choices"][0]["message"]["content"]
        self.co = cod

        self.co = self.co.replace("\n#\n\n", '#').replace('#\n', '#').replace('```', '#').replace('\n    ', '\n')
        self.co = self.co.split(':', 1)[-1].split("\n#\n\n")[0]


        # Execute the generated code
        #result_dict = {}
        try:
            import re
            import importlib
            get=self.co.split('\nresult = ')[1]
            re=importlib.import_module("re")
            return eval(get,globals(),{"da":da,"re":re})
            # exec_globals = result_dict
            # #self.da=da
            # exec(self.co, exec_globals, {"da": self.da})
            # result = result_dict.get('result')
            # return result
        except Exception as e:
            print("Error:", e)

    def get_pattern(self):
        return self.co
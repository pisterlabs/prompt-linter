import PromptBuilder
import LLMConfigurator
import openai
import os
from abc import ABCMeta, abstractmethod


class Unit(metaclass=ABCMeta):
    def __init__(self, output_des, promptName="", inputs=[]):
        self.prompt_builder = PromptBuilder.PromptBuilder()
        self.config = LLMConfigurator.Config()
        self.promptName = promptName
        self.output_des = output_des
        self.inputs = inputs
        self.output = []
        self.preunits = []
        self.assertion = []
        self.condition = False
        self.value4Conditon = 0

    # Run the template with the provided parameters
    @abstractmethod
    def AIRun(self, inputs):
        pass

    @abstractmethod
    def Non_AIRun(self, inputs):
        pass
class AI_Unit(Unit):

    def __init__(self, output_des, promptName="", inputs="", output="", preunits=[], assertion="", condition=False, value4Conditon=""):
        super().__init__(output_des, promptName, inputs)
        self.prompt_builder = PromptBuilder.PromptBuilder()
        self.config = LLMConfigurator.Config()
        self.promptName = promptName
        self.output_des = output_des
        self.inputs = inputs
        self.output = output
        self.preunits = preunits
        self.assertion = assertion
        self.condition = condition
        self.value4Conditon = value4Conditon

    def AIRun(self, inputs):
        #avoid "You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY)"
        if os.environ.get("OPENAI_API_KEY") is None:
            print("You need to set the OPEN_AI_SECRET_KEY environment variable in .env to your OpenAI API key.")
            exit(1)
        ready_prompt = self.prompt_builder.createPrompt(self.promptName, inputs)
        self.config.add_to_config("prompt", ready_prompt)
        response = openai.Completion.create(
            engine=self.config.engine,
            prompt=self.config.prompt,
            temperature=float(self.config.temperature),
            max_tokens=int(self.config.max_tokens),
            top_p=float(self.config.top_p),
            frequency_penalty=float(self.config.frequency_penalty),
            presence_penalty=float(self.config.presence_penalty),
            stop=self.config.stop_strs
        )
        output = response["choices"][0]["text"]
        # 只选取第一个生成的内容
        output = output.split("=================================")[0]

        # 删除空值之后的内容
        output = output.split("\n")
        # delete the empty string in the list
        output = [i for i in output if i != '']
        output = "\n".join(output)
        a = 1
        # join函数的作用是把list中的元素用指定的字符连接起来
        # 只有当list中存在两个或两个以上的元素时，才会用到指定的字符
        print("---------------------------------------------------------------------")
        print(output)
        print("---------------------------------------------------------------------")
        return output

    def Non_AIRun(self, input):
        pass

class Non_AI_Unit(Unit):
    def __init__(self, output_des, promptName="", inputs="", output="", preunits=[], assertion="", condition=False,
                 value4Conditon="", ): # 实际的输出和预期的输出
        super().__init__(output_des, promptName, inputs)
        self.prompt_builder = PromptBuilder.PromptBuilder()
        self.config = LLMConfigurator.Config()
        self.promptName = promptName
        self.output_des = output_des
        self.inputs = inputs
        self.output = output
        self.preunits = preunits
        self.assertion = assertion
        self.condition = condition
        self.value4Conditon = value4Conditon

    def AIRun(self, input):
        pass

    def Non_AIRun(self, inputs):
        import_FQN = []
        for key, value in inputs.items():
            if key == "FQN":
                value = value.split("\n")
                for i in range(len(value)):
                    if "()" in value[i].split(".")[-1] or "[]" in value[i].split(".")[-1]:
                        # 拿出FQN的最后一个元素
                        token = value[i].split(".")[-1]
                        # 如果token[0]是大写字母，说明是类名
                        if token[0].isupper():
                            # 如果是大写字母，说明是类名
                            value[i] = value[i][:-2]
                        else:
                            # 如果是小写字母，说明是函数名
                            value[i] = value[i].split(".")
                            value[i] = value[i][:-1]
                            value[i] = ".".join(value[i])
                    FQN = "import " + value[i] + ";\n"
                    if FQN not in import_FQN:
                        import_FQN.append(FQN)
                import_FQN = "".join(import_FQN)
        with_FQN_Code = import_FQN + "\n\n" + inputs["code"]
        print(with_FQN_Code)
        return with_FQN_Code

from langchain.llms import OpenAI


class Planner():
    def __init__(
        self,
        base_model,
        ):
        self.base_model = base_model

    def run(self, input):
        print("> Generalize the input task.")

        prompt = """
        Please generalize the following sentences by replacing any numerical or other parts of the sentences with letters.
        Output is only the text after rewriting.

        ------------
        {input_}
        ------------
        """.format(input_ = input)

        responese = self.base_model(prompt)

        print('\033[32m' + f'Generalized Task：{responese}\n' + '\033[0m')

        return responese
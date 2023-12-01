from word2bert import Word2Bert
import openai

API_KEY = "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"
openai.api_key = API_KEY


class titleContext:
    def __init__(self, headline, context):
        self.headline = headline
        self.context = context
        self.w2vSim = 0
        self.bertSim = 0

    def check_similarity(self):
        self.w2vSim, self.bertSim = Word2Bert(self.headline, self.context).run()
        return self.w2vSim, self.bertSim

    def check_similarity2(self):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"i have two sentences \nsentence1 = {self.headline} \nsentence2 = {self.context} \n dont consider additional information, is the second statement true based on first statement? yes or no, why",
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response_text = response.to_dict()['choices'][0]['text'].replace("\n", "")
        print(response_text)
        final_response = response_text[:4]
        if final_response.lower().find('yes') != -1:
            return "YES"
        else:
            return "NO"

    def run(self):
        # print(self.check_similarity())
        print(self.check_similarity2())




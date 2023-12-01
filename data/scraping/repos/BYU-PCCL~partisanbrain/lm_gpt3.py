from lmsampler_baseclass import LMSamplerBaseClass
import openai


class LM_GPT3(LMSamplerBaseClass):
    def __init__(self, model_name):
        '''
        Supported models: 'ada', 'babbage', 'curie', 'davinci', 'gpt3-ada', gpt3-babbage', gpt3-curie', gpt3-davinci'
        '''
        super().__init__(model_name)
        if 'gpt3' in model_name:
            # engine is all text after 'gpt3-'
            self.engine = model_name.split('-')[1]
        else:
            self.engine = self.model_name
        # make sure engine is a valid model
        if self.engine not in ["ada", "babbage", "curie", "davinci"]:
            raise ValueError("Invalid model name. Must be one of: 'ada', 'babbage', 'curie', 'davinci'")
        # make sure API key is set
        if openai.api_key is None:
            raise ValueError("OpenAI API key must be set")


    def send_prompt(self, prompt, n_probs=100):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=1,
            logprobs=n_probs,
        )
        logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
        # sort dictionary by values
        sorted_logprobs = dict(sorted(logprobs.items(), key=lambda x: x[1], reverse=True))
        return sorted_logprobs

    def sample_several(self, prompt, temperature=0, n_tokens=10):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=n_tokens,
            temperature=temperature,
        )
        return response['choices'][0]['text']

if __name__ == '__main__':
    # test LM_GPT2
    lm = LM_GPT3("gpt3-ada")
    # probs = lm.send_prompt("What is the capital of France?\nThe capital of France is")
    text = lm.sample_several(prompt="What is the capital of France?\nThe capital of France is", temperature=0, n_tokens=50)
    print(text)
    pass
from model.gpt.gpt_chat import GPT_turbo_model
from model.anthropic.llm_model import AnthropicModel

class ClosedModelsComparison():
    def __init__(self):
        self.anthropic_model = AnthropicModel(cfg={
            'model_name': 'claude-2',
            'max_output_tokens': 300
        })
        self.gpt4_model = GPT_turbo_model(
            cfg={}
        )



    def test_models(self):
        #change this prompt to test for different questions
        prompt = "How can humans create breathable atmosphere on Mars? And how about cosmic radiation - is it safe for them to live there?"
        answers = self.anthropic_model.generate([prompt])
        print('anthropic answer:')
        print(answers)
        message = [{"role": "user", "content": prompt}]
        answers = self.gpt4_model.generate([message])
        print('GPT4 answer:')
        print(answers)


if __name__ == '__main__':
    models_comparison = ClosedModelsComparison()
    #lets run function test models
    models_comparison.test_models()
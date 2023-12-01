import guidance
import transformers
path = 'vicuna-7b'
# mpt = guidance.llms.transformers.MPTChat('mosaicml/mpt-7b-chat', device=1)
vicuna = guidance.llms.transformers.Vicuna(path, device='mps')
# chatgpt = guidance.llms.OpenAI("gpt-3.5-turbo")

program = guidance('''The best thing about the beach is {{~gen 'best' temperature=0.7 max_token\
s=7}}''', stream=True)
print(program(llm=vicuna))

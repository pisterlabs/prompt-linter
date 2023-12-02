from DollyClientProxy import DollyClientProxy
from langchain import PromptTemplate

ai = DollyClientProxy()
template = "### Instruction:\nClassify these animals {text} into {classification}\n\nExample:\nElephant - Mammal, Crocodile - Reptile, Tiger - Mammal \n\n### Response:\n"
prompt_template = PromptTemplate(template=template, input_variables=["text","classification"])
print(ai.prompt_generate
      (prompt_template.format(text=f'Elephant, Snake, Zebra, Tiger, Catfish, Parrot, \
                                                Gorilla, Crocodile, Lizard', classification='Mammal, Reptile, Bird, Fish'), 
                                                max_tokens=128, top_k=500, temperature=1, do_sample=False) )

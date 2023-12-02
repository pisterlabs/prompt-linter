
import openai

texts = [
"We report the development of GPT-4, a large-scale, multimodal model which can accept"
  "image and text inputs and produce text outputs. While less capable than humans in"
  "many real-world scenarios, GPT-4 exhibits human-level performance on various professional"
  "and academic benchmarks, including passing a simulated bar exam with a score around"
  "the top 10% of test takers.",
  "While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level"
  "performance on various professional and academic benchmarks, including passing a"
  "simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-"
  "based model pre-trained to predict the next token in a document."
]



res = openai.Embedding.create(input = texts, model='text-embedding-ada-002')['data']

print(len(res[0]['embedding']))
print(len(res[1]['embedding']))

"""
1536
1536
"""
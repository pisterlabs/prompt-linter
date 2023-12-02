from langchain import PromptTemplate

similar_word_template = """
I'm going to give you a word, and you need to give me a related word.

My word: {word}
Your word:
"""

prompt = PromptTemplate(
  input_variables=["word"],
  template=similar_word_template,
)

formatted_prompt = prompt.format(word="hello")

print(formatted_prompt)
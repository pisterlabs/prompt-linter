import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

from langchain.chains import LLMBashChain

# from langchain.llms import OpenAI
# llm = OpenAI(temperature=0)

text = "Please write a bash script that prints 'Hello World' to the console."

bash_chain = LLMBashChain.from_llm(llm, verbose=True)

bash_chain.run(text)


from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm_bash.prompt import BashOutputParser

_PROMPT_TEMPLATE = """If someone asks you to perform a task, your job is to come up with a series of bash commands that will perform the task. There is no need to put "#!/bin/bash" in your answer. Make sure to reason step by step, using this format:
Question: "copy the files in the directory named 'target' into a new directory at the same level as target called 'myNewDirectory'"
I need to take the following actions:
- List all files in the directory
- Create a new directory
- Copy the files from the first directory into the second directory
```bash
ls
mkdir myNewDirectory
cp -r target/* myNewDirectory
```

Do not use 'echo' when writing the script.

That is the format. Begin!
Question: {question}"""

PROMPT = PromptTemplate(input_variables=["question"], template=_PROMPT_TEMPLATE, output_parser=BashOutputParser())

bash_chain = LLMBashChain.from_llm(llm, prompt=PROMPT, verbose=True)

text = "Please write a bash script that prints 'Hello World' to the console."

bash_chain.run(text)





from langchain.utilities.bash import BashProcess


persistent_process = BashProcess(persistent=True)
bash_chain = LLMBashChain.from_llm(llm, bash_process=persistent_process, verbose=True)

text = "List the current directory then move up a level."

bash_chain.run(text)


# Run the same command again and see that the state is maintained between calls
bash_chain.run(text)
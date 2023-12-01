from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI


# if getattr(sys, 'frozen', False):
#     script_location = pathlib.Path(sys.executable).parent.resolve()
# else:
#     script_location = pathlib.Path(__file__).parent.resolve()
load_dotenv(dotenv_path=".env")

prompt_template = """Please use Python to implement the following requirements. Only generate code blocks in Markdown format without requiring any additional content. As follows:
```python
print("Hello world!")
```
Requirements:
{user_inputs}

"""
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.9,
    verbose=True,
)
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template),
    verbose=True,
)
user_inputs = input("Input your requirements: ")
res = llm_chain.predict(user_inputs=user_inputs)
print(res)
code_sta = res.find("```python")
code_end = res.rfind("```")
if code_sta != -1 and code_end != -1:
    code = res[(code_sta + 9) : code_end]
    try:
        exec(code)
    except Exception as err:
        print(err)

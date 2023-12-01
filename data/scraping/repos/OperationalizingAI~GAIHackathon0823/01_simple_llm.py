from langchain.llms import OpenAI, Ollama
import dotenv
from pretty_print_callback_handler import PrettyPrintCallbackHandler

dotenv.load_dotenv()


from langchain import PromptTemplate

# prompt_template = PromptTemplate.from_template(
#    "Tell me a {adjective} joke about {content}."
# )
# prompt = prompt_template.format(adjective="lame", content="chickens")
prompt = "What is DevOps ? Please be verbose and detailed in your answer."
llm = OpenAI()
pretty_callback = PrettyPrintCallbackHandler()
llm.callbacks = [pretty_callback]
# llm = Ollama(model="llama2-uncensored")
result = llm(prompt)

print(result)

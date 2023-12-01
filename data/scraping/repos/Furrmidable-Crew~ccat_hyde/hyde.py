import json
import langchain
from cat.mad_hatter.decorators import hook
from cat.log import log


with open("cat/plugins/ccat_hyde/settings.json", "r") as json_file:
    settings = json.load(json_file)


@hook(priority=1)
def cat_recall_query(user_message, cat):

    # Make a prompt from template
    hypothesis_prompt = langchain.PromptTemplate(
        input_variables=["input"],
        template=settings["hyde_prompt"]
    )

    # Run a LLM chain with the user message as input
    hypothesis_chain = langchain.chains.LLMChain(prompt=hypothesis_prompt, llm=cat._llm)
    answer = hypothesis_chain(user_message)
    log(answer, "INFO")
    return answer["text"]

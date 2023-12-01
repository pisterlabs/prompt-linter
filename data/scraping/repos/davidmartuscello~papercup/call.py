from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

def truncate_string(input_string, max_tokens):
    tokens = nltk.word_tokenize(input_string)

    if len(tokens) <= max_tokens:
        return input_string

    truncated_tokens = tokens[-max_tokens:]
    truncated_string = ' '.join(truncated_tokens)

    return truncated_string


#### Context Retrieval

# embedding = create_embedding(user_input)
# context_string = get_context(embedding) #use embedding to search wiki info to create context

def get_context(input_string):
    retrieved_context = "To run a spark job, you must press the big RED button that says 'DO NOT PRESS'"
    # retrieved_context = "This is not any relevant context. I really dont have any information to provide."
    prepared_context = truncate_string(retrieved_context, context_size/2)
    return prepared_context


if __name__ == "__main__":

    context_size = 2048
    history_string = ""
    role_selection = "data engineer"
    user_input = "How to execute a spark job?"

    #### Prompt Templating
    question_template = """
    {history}
    
    Pretend that you are a expert {role}. Answer the question soley based on the context below. If the
    question cannot be answered using the information provided in the context. Simply provide the response "I don't know".
    
    Context: {context}
    
    Question: {query}
    
    Answer: Let's work this out in a step by step way, using only information from the context above.
    """

    question_prompt_template = PromptTemplate(
        input_variables=["history", "role", "query", "context"],
        template=question_template
    )

    import nltk
    # nltk.download('punkt')


    prompt_dict = {
        "history":history_string,
        "role":role_selection,
        "query":user_input,
        "context":get_context(user_input)
    }
    print(question_prompt_template.format(**prompt_dict))

    ##### Query Model: Open AI
    with open("../david_openai_api_key.txt", "r") as f:
        key = f.read().strip()
        llm = OpenAI(openai_api_key=key, temperature=0.9)

    chain = LLMChain(llm=llm, prompt=question_prompt_template)

    print(chain.run(prompt_dict))
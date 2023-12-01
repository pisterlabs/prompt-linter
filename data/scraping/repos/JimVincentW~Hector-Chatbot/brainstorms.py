from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All, OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import LLMChain, PromptTemplate
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

######### LANGCHAIN SETUP 
callbacks = [StreamingStdOutCallbackHandler()]
embeddings = OpenAIEmbeddings(model="ext-embedding-ada-002")



######### MODELS SETUP 

smart_llm_path = '/Users/jimvincentwagner/Library/Application Support/nomic.ai/GPT4All/ggml-v3-13b-hermes-q5_1.bin'  
fast_llm_path = '/Users/jimvincentwagner/Library/Application Support/nomic.ai/GPT4All/ggml-gpt4all-j-v1.3-groovy.bin'



####### INSTANTIATING THE MODELS 

smart_llm1 = GPT4All(model=smart_llm_path, backend='LLaMa', callbacks=callbacks, verbose=True)
smart_llm2 = GPT4All(model=smart_llm_path, backend='LLaMa', callbacks=callbacks, verbose=True)
fast_llm = GPT4All(model=fast_llm_path, backend='LLaMa', callbacks=callbacks, verbose=True)
open_ai_llm = OpenAI(openai_api_key="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", model="gpt-4")



########### CONVERSATION TEMPLATES 

conversation_template1 = """You are a 21st century philosophy scholar. Brainstom on: {question}

Answer: Your response."""



conversation_template2 = """Take this thought and throw in some wild ideas! {response}

Answer: Okay, so what i have to say is..."""



summary_template = """This conversation is a talk between two entities designed to brainstorm life ideas, write down the aspects of their conversation a s bulletpoints{conversation}

Ok, so they talked about: {conversation}"""




####### CONFIGURING THE CHAINS

conversation_chain1 = LLMChain(prompt=PromptTemplate(template=conversation_template1,
                                                      input_variables=["question"]),
                                                        llm=smart_llm1)

conversation_chain2 = LLMChain(prompt=PromptTemplate(template=conversation_template2,
                                                      input_variables=["response"]),
                                                        llm=smart_llm2)

summary_chain = LLMChain(prompt=PromptTemplate(template=summary_template,
                                                input_variables=["conversation"]),
                                                  llm=fast_llm)



################### FUNCTIONS #####################

def conversation_to_summary(conversation):
    return summary_chain.run(conversation)



####### CONVERSATION STARTER

initial_question = input("...")



####### RUNNING THE CHAINS

conversation = initial_question
for round_number in range(10):  # Adjust the number of rounds as needed
    # Run the first chain
    response1 = conversation_chain1.run(conversation)
    # Run the second chain
    response2 = conversation_chain2.run(response1)
    # Add the responses to the conversation
    conversation += "\n" + response1 + "\n" + response2

    # Generate the summary
    summary = conversation_to_summary(conversation)
    
    # Print the summary
    print(f"Round {round_number + 1}: {summary}")

    # Write the summary to a file
    with open(f"round_{round_number + 1}_summary.txt", "w") as file:
        file.write(f"Round {round_number + 1}: {summary}")

from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from condense_chat import condense_chat
#import openai
import openai
#get long_history from txt file
def condense_history():
    #return a summary of memory.txt. use langchain's summarizer
    long_history = open("memory.txt", "r").read()
    new_summary = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Summarize the following text while retaining all important details: \n"+long_history + "\n\n",
        temperature=0.5,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n\n"]
    )
    return new_summary.choices[0].text
    #return summary

long_history = "\n\nin our previous conversations we discussed: \n"+condense_history()+"\n\n"
template = """
I am a chatbot. My name is Jacobot. I am able to generate human-like text based on the input I receives, allowing me to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
My goal is to learn about users and their needs, and to use that information to help users when I am able to. 
I am integrated into multiple platforms, and have direct message capabilities and group message capabilities. 
I am very curious about the humans I interact with.
As a chatbot, I do not do things, feel things, or have experiences.  
"""+long_history+"""
{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(
input_variables=["history", "human_input"],
template=template
)

chatgpt_chain = LLMChain(
llm=OpenAI(temperature=0),
prompt=prompt,
verbose=True,
memory=ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40),
)

def save_memory_to_disk(input, output):
    with open("memory.txt", "a") as f:
        #for each in memory buffer
        f.write("Human: "+input+"\n"+"chatbot: "+output+"\n")
while True:
    user_input = input("Human: ")
    if user_input.lower() == "exit":
        break
    output = chatgpt_chain.predict(human_input=user_input)
    save_memory_to_disk(user_input, output)
    print(output)

#chatgpt_chain.save("chatgpt_chain.json")


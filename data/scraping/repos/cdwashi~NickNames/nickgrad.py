import os
#import openai
import gradio as gr
import config

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

llm = ChatOpenAI(temperature=0.9)

prompt1 = ChatPromptTemplate.from_template(
    "What is a funny nickname to describe \
    a person who does or makes {product}?"
)

chain1 = LLMChain(llm=llm, prompt=prompt1)

#Here I'm simply putting few shot examples in the prompt template instead of calling the few shot prompt template as below
#https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples_chat

template = """You are a witty creative writer with the task of helping \
clients come up with humourous nicknames for the jobs they have or their hobby. \
Your job is to write in 25 words or less a witty story for how the following \
male or female input person got their nickname. Pay strict attention to the gender \
of the person in the story and double check that you use the correct pronoun in your story.

<client>: A man who sells apples would be called..

<writer>: Once he sold a whole apple try to a preacher \
and ever since then he's been called "Johnny Preacher Tree"

<client>: A woman who makes underwear would be called..

<writer>: At fifteen years old "Holey Holly" got her nickname, \
because there were always holes in the socks she made!

<client>: A little boy who is scared of pigs

<writer>: He earned the nickname "Pig Panicker" from his classmates, \
because he was so afraid of pigs.

<client>: A young girl who packs pistols at the gun factory

<writer>: She was so skilled at packing those pistols for shipment \
she earned the nickname "Pistol Packing Annie" """
human_template = "{product}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chain2 = LLMChain(llm=llm, prompt=chat_prompt)

overall_simp_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

#overall_simp_chain.run(user_product)

def nickname_generator(product):
    output = overall_simp_chain.run(product)
    
    return output

testgr = gr.Interface(fn=nickname_generator, 
                      inputs=[gr.Textbox(label="Product Made or Job Held by Person the Nickname is For")], 
                      outputs=[gr.Textbox(label="Witty Nickname and Origin Story")],
                      title="🤖  NICKNAMER😊: The Nickname Generator! by IST Group - AI. Powered by OpenAI.  🤖",                      
                      description="Enter a product or job and get a witty nickname and origin story! 🙄",
                      examples=[["dog walker"], ["a person who makes miniature cars"], ["a person who fixes broken computers"]],
                      allow_flagging="never"
                      )
testgr.launch(share=True)

gr.close_all()

import os
from re import template
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

information = """
emma Rose Roberts (born February 10, 1991)[1] is an American actress. Known for her work in film and television projects of the horror and thriller genres, she has received various accolades, including a Young Artist Award, an MTV Movie & TV Award, and a ShoWest Award.

After making her acting debut in the crime film Blow (2001), Roberts gained recognition for her lead role as Addie Singer on the Nickelodeon television teen sitcom Unfabulous (2004–2007). For the series, she released her debut soundtrack album, Unfabulous and More, in 2005. She went on to appear in numerous films, including Aquamarine (2006), Nancy Drew (2007), Wild Child (2008), Hotel for Dogs (2009), Valentine's Day (2010), It's Kind of a Funny Story (2010), and The Art of Getting By (2011).

Looking for more mature roles, Roberts obtained starring roles in the films Lymelife (2008), 4.3.2.1. (2010), Scream 4 (2011), Adult World (2013), We're the Millers (2013), Palo Alto (2013), The Blackcoat's Daughter (2015), Nerve (2016), Who We Are Now (2017), Paradise Hills (2019), and Holidate (2020). Roberts gained further recognition for her starring roles in multiple seasons of the FX anthology horror series American Horror Story (2013–present) and for the lead role of Chanel Oberlin on the Fox comedy horror series Scream Queens (2015–2016).[2]
"""


if __name__ == "__main__":
    print("hello langchain")

    print(os.environ["OPENAI_API_KEY"])

    summary_template = """
        given the information {information} about a person from Iwant you to create:
        1- a short summary
        2- two intereting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(information=information))

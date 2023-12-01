import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

load_dotenv()

information = """Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa's administrative capital.
He has British and Pennsylvania Dutch ancestry. His mother, Maye Musk (née Haldeman), is a model and dietitian born in
Saskatchewan, Canada, and raised in South Africa. His father, Errol Musk, is a South African electromechanical engineer,
pilot, sailor, consultant, and property developer, who partly owned a Zambian emerald mine near Lake Tanganyika, as well
as a rental lodge at the Timbavati Private Nature Reserve. Musk has a younger brother, Kimbal, and a younger sister,
Tosca.

Musk's family was wealthy during his youth. His father was elected to the Pretoria City Council as a representative of
the anti-apartheid Progressive Party and has said that his children shared their father's dislike of apartheid.
His maternal grandfather and namesake, Joshua Elon Haldeman, was an American-born Canadian who took his family on
record-breaking journeys to Africa and Australia in a single-engine Bellanca airplane. Haldeman was a member of the
Social Credit Party of Canada, possessed antisemitic beliefs, and supported the Technocracy movement.
After his parents divorced in 1980, Musk chose to live primarily with his father. Musk later regretted his decision and
became estranged from his father. He has a paternal half-sister and a half-brother.

Musk was often bullied. In one incident, after calling a boy whose father had committed suicide "stupid",
Musk was severely beaten and thrown down concrete steps. He was also an enthusiast reader of books, attributing
his success in part to having read Benjamin Franklin: An American Life, Lord of the Flies, the Foundation series,
and The Hitchhiker's Guide to the Galaxy. At age ten, he developed an interest in computing and video games, teaching
himself how to program from the VIC-20 user manual. At age twelve, Musk sold his BASIC-based game Blastar to PC and
Office Technology magazine for approximately $500.
"""

if __name__ == "__main__":
    print("Hello, Langchain!")

    print(f"OpenAI API key: {os.environ.get('OPENAI_API_KEY')}")

    summary_template = """
    Given the information {information} about a person from I want you to create :
        1. a short summary
        2. two interesting facts about them
    """

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", summary_template),
        ]
    )

    # chat_prompt.format_messages(information=information)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

    chain = chat_prompt | llm

    res = chain.invoke({"information": information})

    print(res.content)

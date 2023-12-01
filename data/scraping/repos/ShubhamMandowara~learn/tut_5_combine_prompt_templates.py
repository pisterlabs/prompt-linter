
from langchain.prompts.chat import ChatPromptTemplate


if __name__ == "__main__":
    template = 'Suggest me company names in {input_language}'
    human_template ='{text}'

    chat_prompt = ChatPromptTemplate.from_messages([
        ('system', template),
        ('human', human_template)
    ])
    print(chat_prompt)
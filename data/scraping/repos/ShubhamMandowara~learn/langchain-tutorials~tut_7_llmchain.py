from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt


if __name__ == "__main__":
    load_dotenv()

    prompt = load_prompt('simple_prompt.yaml')

    model = ChatOpenAI()

    chain = LLMChain(llm=model, prompt=prompt)

    response = chain.run(input_language='English', user_text='A company that creates youtube videos on AI')
    print(response)
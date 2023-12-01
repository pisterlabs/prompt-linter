import os

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory

def Details(apikey, brand, name, score, orig_price, discount_price, cust_summary, size, delivery):
    os.environ['OPENAI_API_KEY'] = apikey
    llm = OpenAI(temperature=0.5)

    template = """You are a helpful shopping agent that creates a summary that matches an input.
        - Input will be details of clothes.
        - Output must be only the summary of the input. Do not add any description about the output.

        ```TypeScript
        interface Detail {
            brand: string;
            name: string;
            score: number;
            orig_price: string;
            discount_price: string;
            cust_summary: (age: string, sex: string) => string;
            size: {
                opt: string;
                length: number;
                shoulder: number;
                chest: number;
                arm: number;
            };
            delivery: string;
        }
        ```

        Based on the given Detail data, you have to summary the details for online shopping customers.

        Relavant Information: 
        {history}

        Conversation:
        input: {details}
        summary: 이 상품은 ${brand}에서 나온 ${name}입니다. ${cust_summary}에게 인기가 많으며 평점은 ${score}입니다. 정가는 ${orig_pirce}이지만, 회원 등급에 따라 ${discount_price}에 구매하실 수 있습니다. 지금 상품을 구매하시면 ${delivery}입니다. 사이즈 정보는 아래와 같습니다. [${size.opt} - 총장(${size.length}cm), 어깨너비(${size.shoulder}cm), 가슴단면(${size.chest}cm), 소매길이(${size.arm}cm)]"""


    prompt = PromptTemplate(
        input_variables=["history", "input"], 
        template=template
    )

    memory = ConversationKGMemory(llm=llm)
    memory.save_context({"input":"..."}, {"output":"..."})
    memory.save_context({"input":"..."}, {"output":"..."})

    conversation_with_kg = ConversationChain(
        llm=llm,
        verbose=True,
        prompt=prompt,
        memory=memory
    )

    return conversation_with_kg.predict(input=user_input).replace('"', '')
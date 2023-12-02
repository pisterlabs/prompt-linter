from decouple import config
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

from naturalchain.tools.calculator.tool import PythonCalculatorTool
from naturalchain.tools.sign_tx.tool import SignTransactionTool

OPENAI_API_KEY = config("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)  # type: ignore

if __name__ == "__main__":
    tools = [
        SignTransactionTool(),
        PythonCalculatorTool(),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    recipientAddress = "0x553447655796B835D04b6d432A2Bf426FE3bF559"
    ether_amount = 0.001
    token_amount = 10

    erc20_token_address = "0xde9e5f071cc331690fd776d224024152302afa22"
    question = f"Transfer {token_amount} tokens to {recipientAddress}, using this erc20 contract: ${erc20_token_address} on Sepolia. Give me the resulting transaction hash"
    #question = f"Send a transaction of {ether_amount} ethers to {recipientAddress} on Sepolia. Give me the resulting transaction hash"
    ### D:question = f"Use mint method to mint {token_amount} tokens to {recipientAddress}, using this erc20 contract: ${erc20_token_address} on Ethereum mainnet. Give me the resulting transaction hash"

    response = agent.run(question)

    print(response)

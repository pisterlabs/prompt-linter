from decouple import config
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

from naturalchain.tools.calculator.tool import PythonCalculatorTool
from naturalchain.tools.rpc.tool import RPCTool

OPENAI_API_KEY = config("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)  # type: ignore

if __name__ == "__main__":
    tools = [
        RPCTool(),
        PythonCalculatorTool(),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    dai_contract = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
    eth_address = "0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326"
    arbitrum_dai_address = "0xA10C7CE4B876998858B1A9E12B10092229539400"

    avalanche_token_contract = "0x3A88ffd737f9194229EeBA4856c8EdBEc0ad8C88"
    avalanche_token_holder = "0x81A2E590D467C9b642bfA93AF76e82a9f78C4e58"
    avalanche_address = "0x3A88ffd737f9194229EeBA4856c8EdBEc0ad8C88"

    question = f"Get the balance for the address {eth_address} on Ethereum mainnet"
    question = f"Get the tokens owned by the address {arbitrum_dai_address} on Ethereum mainnet on the {dai_contract} contract. The token has 18 decimals"
    question = (
        f"Get the balance for the address {avalanche_address} on Avalanche mainnet"
    )

    token_id = 8641
    token_holder = "0xd46C8648F2ac4Ce1A1aace620460fbd24F640853"
    nft_contract = "0xED5AF388653567Af2F388E6224dC7C4b3241C544"
    question = f"Get the owner of the NFT with token id {token_id} on Ethereum mainnet on the {nft_contract} contract"
    question = f"Get how many NFTs are owned by the address {token_holder} on Ethereum mainnet on the {nft_contract} contract"

    response = agent.run(question)

    print(response)

import os, json

from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.chains import create_extraction_chain


def update_accounts():
    with open('config.json') as f:
        config = json.load(f)
    os.environ["GOOGLE_CSE_ID"] = config['GOOGLE_CSE_ID']
    os.environ["GOOGLE_API_KEY"] = config['GOOGLE_API_KEY']

    import main
    from main import app
    from app.models import Account, Interaction, AccountActivity, InteractionType

    search = GoogleSearchAPIWrapper()

    def top1(query):
        return search.results(query, 1)

    # with app.app_context:
    headcount_dict = {}
    all_accounts = Account.query.all()

    for account in all_accounts:
        if account.NumberOfEmployees is None:
            tool = Tool(
                name="Google Search Snippets",
                description="Search Google for recent results.",
                func=top1,
            )
            account_query = tool.run("What is the estimated headcount for: " + account.Name)

            schema = {
                "properties": {
                    "headcount": {"type": "integer"},

                },
                "required": ["headcount"],
            }
            # Run chain
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=config['openai_api-key'])
            chain = create_extraction_chain(schema, llm)
            for headcount in account_query:
                total = chain.run(headcount['snippet'])
                headcount_dict[account.Id] = total
                account.NumberOfEmployees = headcount_dict[account.Id][0]['headcount']
                main.db.session.add(account)
            main.db.session.commit()

    return headcount_dict

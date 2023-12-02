import guidance

guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo")


def first_chat():
    program = guidance(
        """
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
{{conversation_question}}
{{~/user}}

{{! this is a comment. note that we don't have to use a stop="stop_string"}}
{{!~ for the gen command below because Guidance infers the stop string}}
{{#assistant~}}
{{gen "response"}}
{{/assistant}}"""
    )

    executed_program = program(conversation_question="What is the meaning of life")
    print(executed_program)


def multi_step():
    experts = guidance(
        """
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
I want a response to the following question:
{{query}}
Who are 3 world-class experts (past or present) who would be great at answering this?
Please don't answer the question or comment on it yet.
{{~/user}}

{{#assistant~}}
{{gen "experts" temperature=0 max_tokens=300}}
{{~/assistant}}

{{#user~}}
Great, now please answer the question as if these experts had collaborated
in writing a joint anonymous answer.
if the experts would disagree, just present their different positions as alternatives
in the answer itself(e.g. 'some might argue... other might argue...').
Please start your answer with ANSWER:
{{~/user}}

{{#assistant~}}
{{gen "answer" temperature=0 max_tokens=500}}
{{~/assistant}}
"""
    )

    print(experts(query="What is the meaning of life?"))


def with_hidden():
    program = guidance(
        """
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#block hidden=True}}
{{#user~}}
Please tell me a joke
{{~/user}}

{{! note that we don't have guidance controls inside the assistant role because
    the OpenAI API does not yet support that (Transformers chat models do)}}
{{#assistant~}}
{{gen "joke"}}
{{~/assistant}}
{{~/block}}

{{#user~}}
Is the following joke funny? why or why not?
{{joke}}
{{~/user}}

{{#assistant~}}
{{gen "funny"}}
{{~/assistant}}
"""
    )

    print(program())


def agents():
    program = guidance(
        """
    {{#system~}}
    You are a helpful assistant.
    {{~/system}}

    {{~#geneach "conversation" stop=False}}
    {{#user~}}
    {{set "this.user_text" (await "user_text")}}
    {{~/user}}

    {{#assistant~}}
    {{gen "this.ai_text" temperature=0 max_tokens=300}}
    {{~/assistant}}
    {{~/geneach}}
    """
    )

    program = program(user_text="hi there")

    print(program)
    print(program["conversation"])

    program = program(user_text="What is the meaning of life?")
    print(program["conversation"])


def using_tools():
    def is_search(completion):
        return "<search>" in completion

    def search(query):
        return [
            {
                "title": "How do I cancel a Subscription? | Facebook Help Center",
                "snippet": "To stop a monthly Subscription to a creator: Go to the creator's Facebook Page using the latest version of the Facebook app for iOS, Android or from a computer. Select Go to Supporter Hub. Select . Select Manage Subscription to go to the iTunes or Google Play Store and cancel your subscription. Cancel your Subscription at least 24 hours before ...",
            },
            {
                "title": "News | FACEBOOK Stock Price Today | Analyst Opinions - Insider",
                "snippet": "Stock | News | FACEBOOK Stock Price Today | Analyst Opinions | Markets Insider Markets Stocks Indices Commodities Cryptocurrencies Currencies ETFs News Facebook Inc (A) Cert Deposito Arg Repr...",
            },
            {
                "title": "Facebook Stock Price Today (NASDAQ: META) Quote, Market Cap, Chart ...",
                "snippet": "Facebook Stock Price Today (NASDAQ: META) Quote, Market Cap, Chart | WallStreetZen Meta Platforms Inc Stock Add to Watchlist Overview Forecast Earnings Dividend Ownership Statistics $197.81 +2.20 (+1.12%) Updated Mar 20, 2023 Meta Platforms shares are trading... find out Why META Price Moved with a free WallStreetZen account Why Price Moved",
            },
        ]

    search_demo = guidance(
        """Search results:
{{~#each results}}
<result>
{{this.title}}
{{this.snippet}}
</result>{{/each}}"""
    )

    demo_results = [
        {
            "title": "OpenAI - Wikipedia",
            "snippet": "OpenAI systems run on the fifth most powerful supercomputer in the world. [5] [6] [7] The organization was founded in San Francisco in 2015 by Sam Altman, Reid Hoffman, Jessica Livingston, Elon Musk, Ilya Sutskever, Peter Thiel and others, [8] [1] [9] who collectively pledged US$ 1 billion. Musk resigned from the board in 2018 but remained a donor.",
        },
        {
            "title": "About - OpenAI",
            "snippet": "About OpenAI is an AI research and deployment company. Our mission is to ensure that artificial general intelligence benefits all of humanity. Our vision for the future of AGI Our mission is to ensure that artificial general intelligence—AI systems that are generally smarter than humans—benefits all of humanity. Read our plan for AGI",
        },
        {
            "title": "Ilya Sutskever | Stanford HAI",
            "snippet": """Ilya Sutskever is Co-founder and Chief Scientist of OpenAI, which aims to build artificial general intelligence that benefits all of humanity. He leads research at OpenAI and is one of the architects behind the GPT models. Prior to OpenAI, Ilya was co-inventor of AlexNet and Sequence to Sequence Learning.""",
        },
    ]

    s = search_demo(results=demo_results)
    print(s)

    practice_round = [
        {"role": "user", "content": "Who are the founders of OpenAI?"},
        {
            "role": "assistant",
            "content": "<search>Who are the founders of OpenAI</search>",
        },
        {"role": "user", "content": str(search_demo(results=demo_results))},
        {
            "role": "assistant",
            "content": "The founders of OpenAI are Sam Altman, Reid Hoffman, Jessica Livingston, Elon Musk, Ilya Sutskever, Peter Thiel and others.",
        },
    ]

    program = guidance(
        """
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
From now on, whenever your response depends on any factual information,
please search the web by using the function <search>query</search>
before responding. I will then paste web results in, and you can respond.
{{~/user}}

{{#assistant~}}
Ok, I will do that. Let's do a practice round
{{~/assistant}}

{{#each practice}}
{{#if (== this.role "user")}}
{{#user}}{{this.content}}{{/user}}
{{else}}
{{#assistant}}{{this.content}}{{/assistant}}
{{/if}}
{{/each}}

{{#user~}}
That was great, now let's do another one.
{{~/user}}

{{#assistant~}}
Sounds good
{{~/assistant}}

{{#user~}}
{{user_query}}
{{~/user}}

{{#assistant~}}
{{gen "query" stop="</search>"}}{{#if (is_search query)}}</search>{{/if}}
{{~/assistant}}

{{#user~}}
Search results: {{#each (search query)}}
<result>
{{this.title}}
{{this.snippet}}
</result>{{/each}}
{{~/user}}

{{#assistant~}}
{{gen "answer"}}
{{~/assistant}}
"""
    )

    query = "What is Facebook's stock price right now?"
    program = program(
        user_query=query,
        search=search,
        is_search=is_search,
        practice=practice_round,
    )

    print(program)


if __name__ == "__main__":
    # first_chat()
    # multi_step()
    # with_hidden()
    # agents()
    using_tools()

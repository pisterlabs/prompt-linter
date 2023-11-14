from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback


from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


class SummaryChain:
    """
    Class for creating a summary chain for extracting main sentiment and summary from wine reviews.

    Attributes
    ----------
    df : pandas.DataFrame
        The dataframe that contains the wine reviews.
    llm : langchain.chat_models.ChatOpenAI
        The language model for extracting the summary.

    Methods
    -------
    build_chain():
        Builds a SequentialChain for sentiment extraction.
    generate_sequentially():
        Generates sentiment and summary sequentially for each review in the dataframe.
    """

    def __init__(self, df, llm):
        self.df = df
        self.llm = llm

    def build_chain(self):
        """
        Builds a SequentialChain for sentiment extraction.

        Returns
        -------
        tuple
            A tuple containing the built SequentialChain, the output parser, and the response format.
        """

        llm = self.llm

        sentiment_schema = ResponseSchema(
            name="sentiment",
            description="The main sentiment of the review, limited to 3 words.",
        )
        summary_schema = ResponseSchema(
            name="summary",
            description="Brief Summary of the review, limited to one paragraph.",
        )

        sentiment_response_schemas = [sentiment_schema, summary_schema]

        output_parser = StructuredOutputParser.from_response_schemas(
            sentiment_response_schemas
        )
        response_format = output_parser.get_format_instructions()

        ## sentiment and Summary Chain
        sentiment_prompt = ChatPromptTemplate.from_template(
            """Act like an expert somellier. Your goal is to extract the main sentiment from wine reviews, delimited by triple dashes. Limit the sentiment to 3 words. \

            ---
            Review: {review}
            ---

            {response_format}
            """
        )

        sentiment_chain = LLMChain(
            llm=llm, prompt=sentiment_prompt, output_key="sentiment"
        )

        chain = SequentialChain(
            chains=[sentiment_chain],
            input_variables=["review"] + ["response_format"],
            output_variables=["sentiment"],
            verbose=False,
        )

        return chain, output_parser, response_format

    def generate_sequentially(self):
        """
        Generates sentiment and summary sequentially for each review in the dataframe.
        The extracted sentiments, summaries, and costs are added to the dataframe.
        """
        df = self.df
        chain, output_parser, response_format = self.build_chain()
        for _, row in df.iterrows():
            review = row["description"]
            unique_id = row["unique_id"]
            inputs = {
                "review": review,
                "response_format": response_format,
            }
            with get_openai_callback() as cb:
                resp = chain.run(inputs)
                cost = cb.total_cost
            summary = output_parser.parse(resp)["summary"]
            sentiment = output_parser.parse(resp)["sentiment"]
            df.loc[
                df["unique_id"] == unique_id, ["summary", "sentiment", "sentiment_cost"]
            ] = [summary, sentiment, cost]


class CharacteristicsChain:
    """
    Class for creating a chain for extracting top five main characteristics of the wine.

    Attributes
    ----------
    df : pandas.DataFrame
        The dataframe that contains the wine reviews.
    llm : langchain.chat_models.ChatOpenAI
        The language model for extracting the characteristics.

    Methods
    -------
    build_chain():
        Builds a SequentialChain for characteristic extraction.
    generate_sequentially():
        Generates characteristics sequentially for each wine in the dataframe.
    """

    def __init__(self, df, llm):
        self.df = df
        self.llm = llm

    def build_chain(self):
        """
        Builds a SequentialChain for characteristic extraction.

        Returns
        -------
        tuple
            A tuple containing the built SequentialChain, the output parser, and the response format.
        """

        llm = self.llm

        characteristics_schema = []
        for i in range(1, 6):
            characteristics_schema.append(
                ResponseSchema(
                    name=f"characteristic_{i}",
                    description=f"The number {i} characteristic. One or two words long.",
                )
            )

        output_parser = StructuredOutputParser.from_response_schemas(
            characteristics_schema
        )
        response_format = output_parser.get_format_instructions()

        characteristics_prompt = ChatPromptTemplate.from_template(
            """
        Act like an expert somellier. You will receive the name, the summary of the review and the county of origin of a given wine, delimited by triple dashes.
        Your goal is to extract the top five main characteristics of the wine.
            ---
            Wine Name: {wine_name}
            Country: {country}
            Summary Review: {summary}
            ---

            {response_format}

            """
        )
        characteristics_chain = LLMChain(
            llm=llm, prompt=characteristics_prompt, output_key="characteristics"
        )

        chain = SequentialChain(
            chains=[characteristics_chain],
            input_variables=["wine_name", "summary", "country"] + ["response_format"],
            output_variables=["characteristics"],
            verbose=False,
        )

        return chain, output_parser, response_format

    def generate_sequentially(self):
        """
        Generates characteristics sequentially for each wine in the dataframe.
        The extracted characteristics and costs are added to the dataframe.
        """

        df = self.df
        chain, output_parser, response_format = self.build_chain()
        for _, row in df.iterrows():
            summary = row["summary"]
            country = row["country"]
            unique_id = row["unique_id"]
            title = row["title"]
            inputs = {
                "summary": summary,
                "wine_name": title,
                "country": country,
                "response_format": response_format,
            }
            with get_openai_callback() as cb:
                resp = chain.run(inputs)
                cost = cb.total_cost
            characteristics = [
                output_parser.parse(resp)[f"characteristic_{i}"] for i in range(1, 6)
            ]
            df.loc[
                df.unique_id == unique_id,
                [
                    "characteristic_1",
                    "characteristic_2",
                    "characteristic_3",
                    "characteristic_4",
                    "characteristic_5",
                    "cost_characteristics",
                ],
            ] = characteristics + [cost]

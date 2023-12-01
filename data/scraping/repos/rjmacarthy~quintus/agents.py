import guidance
from database.repository import Repository
from database.repository import Repository
from database.schema.document import Document
from utils.encoder import Encoder

gpt_35_turbo = guidance.llms.OpenAI("gpt-3.5-turbo")
text_davinci_003 = guidance.llms.OpenAI("text-davinci-003")


class Agents:
    def __init__(self):
        self.repository = Repository(Document)
        self.encoder = Encoder()

    def get_context(self, question: str) -> str:
        max_prompt_length = 1024
        num_relevant_docs = 3
        embeddings = self.encoder.encode(question)
        results = self.repository.search(embeddings)
        text = " ".join([result.doc_text for result in results[:num_relevant_docs]])
        return text[:max_prompt_length]

    def assistant(self, entity="user", context="", question=""):
        agent = guidance(
            """
            {{#system~}}
                You are a helpful and terse cusrtom support agent you only answer questions that are related to your domain.
            {{~/system}}
            {{#user~}}
                Hi are you there?
            {{~/user}}
            {{#assistant~}}
                Hello, how can I help you today?
            {{~/assistant}}
            {{#user~}}
                You will be given a context and a question. You must answer the question based on the context.
                Continue the conversation as normal.
                Question: {{question}}
                Context: {{context}}
                Do not mention the context in your answer if you cannot find the answer in the context.
                Do not answer questions about anything else other than the context and the question.
            {{~/user}}
            {{#assistant~}}
                {{gen 'reply' temperature=0 max_tokens=300}}
            {{~/assistant}}
        """, llm=gpt_35_turbo
        )
        return agent(entity=entity, context=context, question=question)

    def classifier(self):
        agent = guidance(
            """
            {{#system~}}
                You are a document classifying agent responsible for classifying documents into one of the following options.
                Only answer with one word.
            {{~/system}}
            {{#user~}}
                Classify the following document into one of the following categories:
                Categories: {{categories}}
                Document: {{document}}
            {{~/user}}
            {{#assistant~}}
                {{gen 'reply' temperature=0 max_tokens=300}}
            {{~/assistant}}
        """, llm=gpt_35_turbo
        )
        return agent()

    def json_generator():
        return guidance(
            """
            Your task is to return valid JSON objects for the given data and structure.
            Data could be in any format with separators such as commas, newlines, or spaces or any combination.
            You may receive additional arbitrary data.

            Example:

            Data: [["joe bloggs, male, 38, Wales"], ["jane, "female", 25, England"]]

            Structure:
            ```json
                { "name": string, "gender": string, "age": number, "id": number, "country": string }
            ```

            ```json
            [
                {
                    "name": "Joe Bloggs",
                    "gender": "Male",
                    "age": 38
                    "country": "Wales"
                },
                {
                    "name": "Jane",
                    "gender": "Female",
                    "age": 25,
                    "country": "England"
                }
            ]
            ```

            Please provide the JSON for the following data.
            Data: {{data}}
            {f"Structure: {{structure}}" if structure else ""}
            Arbitrary Data: {{arbitrary_data}}
            {{gen 'data' temperature=0 max_tokens=500}}
        """,
            llm=text_davinci_003,
        )

    def json_cleaner(self):
        return guidance(
            """
            You are a JSON cleaner agent specialized in cleaning JSON from various documents, regardless of their format or quality.
            Your task is to return clean JSON.
            The input document can be anything and may be poorly formatted.

            Please provide the JSON for the following data.
            Data: {{data}}
            {{gen 'data' temperature=0 max_tokens=500}}
        """,
            llm=text_davinci_003,
        )

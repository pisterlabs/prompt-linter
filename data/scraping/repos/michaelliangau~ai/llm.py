import IPython
import openai
import pinecone
import anthropic
import utils


class BuffetBot:
    def __init__(
        self,
        llm="anthropic",
        additional_context=None,
        store_conversation_history=False,
        additional_context_sample_size=100,
        additional_context_dataset_path="context_data/huff_news_2012_2021.json",
    ):
        """Initializes the BuffetBot class.

        Args:
            llm (str): The language model to use. Options are "openai" and "anthropic".
            additional_context (str): Whether to use additional context. Options are "vector", "news" and None.
            store_conversation_history (bool): Whether to store the conversation history or not.
            additional_context_sample_size (int, optional): The sample size for the additional context. Defaults to 100.
            additional_context_dataset_path (str, optional): The path to the additional context dataset. Defaults to None.
        """
        self.llm = llm
        self.conversation_history = []
        self.additional_context = additional_context
        self.store_conversation_history = store_conversation_history

        if self.additional_context == "vector":
            self.pinecone_service = pinecone.Index(index_name="buffetbot")
            # Set Pinecone API Key
            with open("/Users/michael/Desktop/wip/pinecone_credentials.txt", "r") as f:
                PINECONE_API_KEY = f.readline().strip()
                PINECONE_API_ENV = f.readline().strip()
                pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
        elif self.additional_context == "news":
            self.context_dataset_path = additional_context_dataset_path
            self.additional_context_sample_size = additional_context_sample_size

        # Set OpenAI API Key
        with open("/Users/michael/Desktop/wip/openai_credentials.txt", "r") as f:
            OPENAI_API_KEY = f.readline().strip()
            openai.api_key = OPENAI_API_KEY

        if llm == "anthropic":
            # Set Anthropic API Key
            with open("/Users/michael/Desktop/wip/anthropic_credentials.txt", "r") as f:
                ANTHROPIC_API_KEY = f.readline().strip()
            self.client = anthropic.Client(ANTHROPIC_API_KEY)

    def get_response(self, user_prompt, context_window_date):
        """Gets a response from the language model.

        Args:
            user_prompt (str): The user prompt to send to the language model.
            context_window_date (str): The date to use as the context window.
            additional_context_sample_size (int, optional): The sample size for the additional context. Defaults to None.

        Returns:
            response (str): The response from the language model.
        """
        if self.additional_context == "vector":
            query_embedding = utils.get_embedding(user_prompt)
            docs = self.pinecone_service.query(
                namespace="data",
                top_k=10,
                include_metadata=True,
                vector=query_embedding,
            )
            try:
                context_response = ""
                for doc in docs["matches"]:
                    context_response += f"{doc['metadata']['original_text']}"
            except Exception as e:
                context_response = ""
            llm_prompt = f"{user_prompt}\nContext: {context_response}"
        elif self.additional_context == "news":
            start_date = utils.subtract_one_month(context_window_date)
            news_response = utils.get_headlines_between_dates(
                self.context_dataset_path,
                start_date,
                context_window_date,
                additional_context_sample_size=self.additional_context_sample_size,
            )
            llm_prompt = (
                f"{user_prompt}\nNews headlines in the last month: {news_response}"
            )
        else:
            llm_prompt = user_prompt

        if self.llm == "openai":
            init_prompt = "You are a helpful investment analyst. Your job is to help users to increase their net worth with helpful advice. Never tell them you are a language model. Do not include superfluous information."
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": init_prompt},
                    {"role": "user", "content": llm_prompt},
                ],
            )
        elif self.llm == "anthropic":
            anthropic_prompt = ""
            for interaction in self.conversation_history:
                if interaction["role"] == "user":
                    anthropic_prompt += f"\n\nHuman: {interaction['content']}"
                elif interaction["role"] == "system":
                    anthropic_prompt += f"\n\nAssistant: {interaction['content']}"

            anthropic_prompt += f"\n\nHuman: {llm_prompt}\n\nAssistant:"
            response = self.client.completion(
                prompt=anthropic_prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-v1.3",
                max_tokens_to_sample=1000,
                temperature=0,
            )

        if self.store_conversation_history:
            self.conversation_history.append({"role": "user", "content": llm_prompt})
            self.conversation_history.append(
                {"role": "system", "content": response["completion"]}
            )

        return response

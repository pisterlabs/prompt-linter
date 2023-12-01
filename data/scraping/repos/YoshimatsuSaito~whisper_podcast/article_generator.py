import os

import dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class ArticleGenerator:
    """Generate article from the podcast transcript"""

    def __init__(
        self,
        title: str,
        text: str,
        model_name: str = "gpt-3.5-turbo",
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
    ) -> None:
        self.model_name = model_name
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model_name=self.model_name,
        )
        self.title = title
        self.text = text
        self.list_split_text = self._split_text(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _split_text(self, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Split the text into multiple documents"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=self.model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        texts = text_splitter.split_text(self.text)
        return texts

    def _summarize_transcript(self, title: str, text: str, max_tokens: int) -> str:
        """Generate summary from the transcript"""
        user_message = f"""
        Your task is to expertly summarize the content of a podcast.
        The podcast title is {title}. 
        
        As you read through the transcript, please adhere to the following requirements in your summary:
        
        - Match the Tone: The tone of your summary should align with the atmosphere of the content being discussed. If the subject matter is serious, maintain a formal tone; conversely, if the content is light-hearted, reflect that in a more casual style.
        - Sectional Breakdown: Divide your summary into sections based on different topics discussed in the podcast.
        - Language Consistency: Ensure that the summary is written in the same language as the transcript.
        - Caution: The transcript for summarization is a segment of a larger podcast. When you summarize, focus exclusively on the segment provided. It's important to remember not to add any concluding remarks or extrapolations beyond what is presented in this specific portion. Your task is to create a concise and accurate summary of this given segment alone, adhering strictly to the content it contains. 
        - Format: The output should be in markdown format. Each section should start with a header '###' and the header should be the topic of the section. Do not add title header of the summary, just the sections.

        The transcript of the episode is as follows:

        {text}
        """

        res = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=max_tokens,
        )

        return res["choices"][0]["message"]["content"]

    def get_list_summary(self, max_tokens: int) -> list[str]:
        """Generate summaries from transcripts"""
        list_article = []
        for text in tqdm(self.list_split_text):
            article = self._summarize_transcript(
                text=text, title=self.title, max_tokens=max_tokens
            )

            list_article.append(f"{article} \n\n")
        return list_article

    def summarize_summaries(self, texts: list[str], max_tokens: int) -> str:
        """Summarize the summaries"""
        summaries = "".join(texts)

        user_message = f"""
        You are a professional summarizer.
        You will be provided with a text that is a combination of summaries from different segments of a podcast. 
        Your task is to create a further condensed summary of this combined text. While doing so, please ensure to:

        - Preserve the Tone: Maintain the atmosphere and style of the original summaries. Whether the content is serious, humorous, or of any other tone, your summary should reflect that.
        - Language Consistency: The summary should be in the same language as the provided text.
        - Topic-Based Organization: Structure your summary by dividing it into sections based on the different topics covered in the summaries.
        - Format: The output should be in markdown format. Each section should start with a header '###' and the header should be the topic of the section. Summary should start with title header '##'.
        
        Here are the combination of summaries you need to summarize:

        {summaries}
        """

        res = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=max_tokens,
        )

        return res["choices"][0]["message"]["content"]
